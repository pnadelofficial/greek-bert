from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
import contextlib
import math
import torch
import torch.distributed as dist
from dataclasses import asdict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import sys
import argparse
from pathlib import Path

# scripts/ is not a package; make its siblings importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import TrainingConfig, setup_distributed, cleanup_distributed, mlm_masking
from paths import repo_path
from runs import record_result, record_run_provenance, write_metrics


def make_autocast(enabled: bool, bf16: bool):
    """AMP context manager.

    `torch.cuda.amp.autocast()` with no dtype defaults to **fp16**, which then
    requires a GradScaler. On H200/H100/B200 bf16 is both faster and more stable
    and needs no loss scaling, so it is the default here.
    """
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast(
        "cuda", dtype=torch.bfloat16 if bf16 else torch.float16
    )


def mean_over_ranks(value: float, count: float, world_size: int):
    """All-reduce a (sum, count) pair so a logged average is the REAL average.

    The validation loader uses a DistributedSampler, so each rank scores its own
    1/world_size shard. Previously only rank 0's shard was ever logged and the
    sums were never reduced, so "Loss/validation" was 1/8 of the test set and
    `epoch_loss` was rank 0's 1/8 too.
    """
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return value / max(count, 1.0), count
    packed = torch.tensor([value, count], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total, n = packed[0].item(), packed[1].item()
    return total / max(n, 1.0), n

parser = argparse.ArgumentParser(description="Reads in training YAML file.")
parser.add_argument(
    "--config-path",
    dest="config_path",
    required=True,
)
args = parser.parse_args()
config_path = args.config_path

def main():
    # init and check GPUs
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')

    # read in config
    hp_config = TrainingConfig.from_yaml(config_path)

    # Seed masks, samplers and any from-scratch init. Derived per-rank so the
    # eight ranks do not draw identical masking patterns.
    torch.manual_seed(hp_config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hp_config.seed + rank)

    if is_main_process:
        print(f"Training on {world_size} GPUs with mixed precision")
        print(f"Local rank: {local_rank}, Global rank: {rank}")
        print(f"Config hash: {hp_config.config_hash()}  (seed={hp_config.seed})")
        record_run_provenance(
            stage="pretrain",
            config_dict=asdict(hp_config),
            config_hash=hp_config.config_hash(),
            extra={"world_size": world_size},
        )
    
    # output dirs
    if is_main_process:
        os.makedirs(hp_config.checkpoint_dir, exist_ok=True)
        os.makedirs(hp_config.tensorboard_dir, exist_ok=True)
    
    # tensorboard writer, only on main process
    writer = None
    if is_main_process:
        writer = SummaryWriter(log_dir=hp_config.tensorboard_dir)
        print(f"TensorBoard logs will be saved to: {hp_config.tensorboard_dir}")
        print(f"Run: tensorboard --logdir={hp_config.tensorboard_dir}")

    # dataset
    if is_main_process:
        print("Loading dataset...")
    
    tokenized_datasets = load_from_disk(hp_config.tokenized_dataset_path)
    tokenized_datasets = tokenized_datasets.remove_columns(
        [col for col in tokenized_datasets['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
    )
    tokenized_datasets.set_format(type="torch")
    
    if is_main_process:
        print("Dataset info:")
        print(f"  Type: {type(tokenized_datasets['train'])}")
        print(f"  Length: {len(tokenized_datasets['train'])}, type: {type(len(tokenized_datasets['train']))}")
        print(f"  Features: {tokenized_datasets['train'].features}")
        
        # Check a sample
        sample = tokenized_datasets['train'][0]
        print("\nSample data:")
        for key, value in sample.items():
            print(f"  {key}: type={type(value)}, dtype={getattr(value, 'dtype', 'N/A')}")

    # model init
    if is_main_process:
        print("Initializing model...")
    
    if not hp_config.pretrained_model:
        # From-scratch arm: the 50k cased Greek BPE with a ModernBERT backbone.
        # The config's vocab_size MUST be set before the model is built, or the
        # LM head comes out at ModernBERT's 50257 and can never match the 50368
        # token Greek tokenizer (audit item 9).
        tokenizer = AutoTokenizer.from_pretrained(str(repo_path("tokenizers", "modernbert-greek-tokenizer")))
        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
        config.vocab_size = len(tokenizer)
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        model = AutoModelForMaskedLM.from_config(config).to(device)
    else:
        config = AutoConfig.from_pretrained(hp_config.pretrained_model)
        model = AutoModelForMaskedLM.from_pretrained(hp_config.pretrained_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hp_config.pretrained_model)

    vocab_size = len(tokenizer)
    if config.vocab_size != vocab_size:
        raise ValueError(
            f"config.vocab_size={config.vocab_size} != len(tokenizer)={vocab_size}. "
            "The LM head and the tokenizer disagree; every loss number from this "
            "run would be meaningless."
        )

    # Only mutate eos_token for models that cannot generate. BERT has no real eos
    # and aliasing it to [PAD] is harmless; baking eos_token="[PAD]" into a
    # ModernBERT export silently misbehaves in generation-style consumers.
    if getattr(config, "model_type", "") == "bert":
        tokenizer.eos_token = tokenizer.pad_token

    # Dataset is pre-tokenized (see utils.tokenize_and_prepare_mlm, run ahead of
    # time); masks are generated dynamically per step in the training loop.

    # wrap with ddp
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # masking parameters
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    ignore_index = -100

    # creating distributed samplers
    train_sampler = DistributedSampler(
        tokenized_datasets["train"],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=22091997
    ) if world_size > 1 else None

    valid_sampler = DistributedSampler(
        tokenized_datasets["test"],
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=22091997
    ) if world_size > 1 else None

    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        batch_size=hp_config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=hp_config.num_workers,
        pin_memory=True,
        persistent_workers=hp_config.num_workers > 0,
        prefetch_factor=hp_config.prefetch_factor if hp_config.num_workers > 0 else None,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        tokenized_datasets["test"], 
        batch_size=hp_config.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=hp_config.num_workers,
        pin_memory=True,
        persistent_workers=hp_config.num_workers > 0,
        prefetch_factor=hp_config.prefetch_factor if hp_config.num_workers > 0 else None,
    )

    # Optimizer with RoBERTa-style hyperparameters:
    # - AdamW with weight decay
    # - β₂=0.98 instead of default 0.999 for better stability with large batches
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(hp_config.lr),
        weight_decay=float(hp_config.weight_decay),
        betas=(0.9, 0.98)  # RoBERTa: β₂=0.98 for stability
    )
    
    # amp scaler/scheduler.
    # bf16 needs no loss scaling at all, so the scaler exists only on the fp16
    # fallback path. Keeping a GradScaler around under bf16 costs a pointless
    # unscale_() pass over every parameter each step.
    use_fp16 = hp_config.use_mixed_precision and not hp_config.bf16
    scaler = GradScaler() if use_fp16 else None
    
    num_training_steps = hp_config.num_epochs * int(len(train_dataloader)) // hp_config.gradient_accumulation_steps
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=float(hp_config.max_lr),
        pct_start=float(hp_config.pct_start),
        total_steps=int(num_training_steps)
    )
    
    # begin training
    ## logging
    if is_main_process:
        print(f"\nStarting training for {hp_config.num_epochs} epochs")
        print(f"Total training steps: {num_training_steps}")
        print(f"Steps per epoch: {len(train_dataloader)}")
        print(f"Effective batch size: {hp_config.batch_size * world_size * hp_config.gradient_accumulation_steps}")
        print(f"Mixed precision: {hp_config.use_mixed_precision} "
              f"({'bf16, no loss scaling' if hp_config.bf16 else 'fp16 + GradScaler'})")
        print(f"Gradient accumulation steps: {hp_config.gradient_accumulation_steps}")
        print(f"Masking: p={hp_config.mask_prob} with 80/10/10 [MASK]/random/unchanged")
        # Report the LR the SCHEDULER will actually apply, not the config value.
        # `lr` and `max_lr` were both 6e-4 in configs/train.yaml, which made the
        # `lr` knob dead while docs/archive/ROBERTA_COMPARISON.md described
        # "base LR scaled by warmup".
        print(f"LR: config lr={hp_config.lr} max_lr={hp_config.max_lr} "
              f"-> scheduler step-0 LR={lr_scheduler.get_last_lr()[0]:.3e}")
        if abs(float(hp_config.max_lr) / max(float(hp_config.lr), 1e-12) - 1.0) < 1e-9:
            print("  NOTE: lr == max_lr, so OneCycleLR ignores `lr` entirely; "
                  "`max_lr` is the peak LR.")

    # Peak LR is what the scheduler actually holds during the decay phase.
    peak_lr = max(lr_scheduler.get_last_lr())
    effective_batch = hp_config.batch_size * world_size * hp_config.gradient_accumulation_steps
    write_metrics(
        {
            "total_training_steps": int(num_training_steps),
            "steps_per_epoch": int(len(train_dataloader)),
            "effective_batch_size": int(effective_batch),
            "peak_lr": peak_lr,
            "mask_prob": float(hp_config.mask_prob),
            "bf16": bool(hp_config.bf16),
            "seed": int(hp_config.seed),
            "world_size": int(world_size),
        },
        stage="pretrain",
    )

    # LR sanity: log the real scheduler LR at fixed early steps so a mis-set
    # warmup/peak is visible in the log instead of inferred from a loss curve.
    lr_probe_steps = {1, 10, 100, 1000}
    
    global_step = 0
    model.train()

    # Best-checkpoint tracking. `improvement` is the last epoch's val-loss drop;
    # if it is still large at the end of the run, the run stopped early.
    best_val_loss = float("inf")
    best_epoch = 0
    improvement = float("inf")
    convergence_threshold = 0.005
    epoch_history: list[dict] = []
    
    for epoch in range(hp_config.num_epochs):
        if is_main_process:
            print(f"\nEpoch {epoch + 1}/{hp_config.num_epochs}")
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(train_dataloader, disable=not is_main_process, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(iterator):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            # Dynamic masking - generate fresh masks every forward pass (RoBERTa approach)
            masked_input_ids, masked_labels = mlm_masking(
                input_ids, 
                mask_token_id=mask_token_id,
                mask_prob=hp_config.mask_prob,
                pad_token_id=pad_token_id,
                ignore_index=ignore_index,
                vocab_size=vocab_size,
                cls_token_id=tokenizer.cls_token_id,
                sep_token_id=tokenizer.sep_token_id,
            )
            masked_input_ids = masked_input_ids.to(device, non_blocking=True)
            masked_labels = masked_labels.to(device, non_blocking=True)

            # amp forward pass
            with make_autocast(hp_config.use_mixed_precision, hp_config.bf16):
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=masked_labels)
                loss = outputs.loss / hp_config.gradient_accumulation_steps

            epoch_loss += loss.item() * hp_config.gradient_accumulation_steps
            num_batches += 1
            
            # amp backward pass
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # update after grad accumulation
            if (step + 1) % hp_config.gradient_accumulation_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process and global_step in lr_probe_steps:
                    probed = lr_scheduler.get_last_lr()[0]
                    print(f"[lr-probe] step {global_step}: {probed:.3e}")
                    writer.add_scalar('Learning_rate_probe', probed, global_step)
            
            # logging
                if is_main_process and global_step % hp_config.log_interval == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    writer.add_scalar('Loss/train', loss.item() * hp_config.gradient_accumulation_steps, global_step)
                    writer.add_scalar('Learning_rate', current_lr, global_step)
                    
                    iterator.set_postfix({
                        'loss': f'{loss.item() * hp_config.gradient_accumulation_steps:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })

        # end of epoch. Reduce across ranks so the logged epoch loss is the
        # whole-corpus average, not rank 0's 1/world_size shard.
        avg_epoch_loss, _ = mean_over_ranks(epoch_loss, num_batches, world_size)
        if is_main_process:
            print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
            
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(hp_config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                model_to_save = model.module if isinstance(model, DDP) else model
            
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
                    'loss': avg_epoch_loss,
                    'global_step': global_step,
                }, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
        
        # validation
        if is_main_process:
            print("Running validation...")
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        # Token-level accuracy: count positions where the argmax over the LM head
        # equals the label. MLM accuracy/PPL — not just the cross-entropy — is
        # what the downstream tasks actually track.
        correct_tokens = 0.0
        labelled_tokens = 0.0
        with torch.no_grad():
            val_iterator = tqdm(valid_dataloader, disable=not is_main_process, desc="Validation")
            for batch in val_iterator:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                # Same masking recipe as training, so val loss is comparable to
                # train loss and to other runs.
                masked_input_ids, masked_labels = mlm_masking(
                    input_ids, 
                    mask_token_id=mask_token_id,
                    mask_prob=hp_config.mask_prob,
                    pad_token_id=pad_token_id,
                    ignore_index=ignore_index,
                    vocab_size=vocab_size,
                    cls_token_id=tokenizer.cls_token_id,
                    sep_token_id=tokenizer.sep_token_id,
                )
                masked_input_ids = masked_input_ids.to(device, non_blocking=True)
                masked_labels = masked_labels.to(device, non_blocking=True)

                with make_autocast(hp_config.use_mixed_precision, hp_config.bf16):
                    outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=masked_labels)
                
                val_loss += outputs.loss.item()
                val_batches += 1

                preds = outputs.logits.argmax(dim=-1)
                hit = (preds == masked_labels) & (masked_labels != ignore_index)
                correct_tokens += hit.sum().item()
                labelled_tokens += (masked_labels != ignore_index).sum().item()

        # Reduce so the number is the real epoch average over the FULL test set,
        # not rank 0's shard (audit item 6). This value is also what selects the
        # best checkpoint, so it has to be a whole-set number on every rank.
        avg_val_loss, _ = mean_over_ranks(val_loss, val_batches, world_size)
        val_acc, _ = mean_over_ranks(correct_tokens, labelled_tokens, world_size)
        val_ppl = math.exp(min(avg_val_loss, 50.0))

        if is_main_process:
            print(f"Validation Loss: {avg_val_loss:.4f}  PPL: {val_ppl:.3f}  masked-token acc: {val_acc:.4f}")
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Eval/val_ppl', val_ppl, epoch)
            writer.add_scalar('Eval/val_masked_token_acc', val_acc, epoch)

        # --- best-checkpoint selection (audit item 5) -----------------------
        # Previously final_model.pt was saved unconditionally and there was no
        # best-by-val-loss checkpoint at all, so "pick the best epoch" did not
        # exist. Every rank writes best_model.pt (identical weights under DDP) so
        # the file is present even if the run is interrupted on rank 0.
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            if is_main_process:
                best_path = os.path.join(hp_config.checkpoint_dir, "best_model.pt")
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
                        'loss': avg_epoch_loss,
                        'val_loss': avg_val_loss,
                        'val_ppl': val_ppl,
                        'val_acc': val_acc,
                        'global_step': global_step,
                    },
                    best_path,
                )
                print(f"  * new best val loss {avg_val_loss:.4f} "
                      f"(-{improvement:.4f}) -> {best_path}")
        
        model.train()

        if is_main_process:
            epoch_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
                'val_acc': val_acc,
                'lr': lr_scheduler.get_last_lr()[0],
            })
            write_metrics(
                {
                    'epochs': epoch_history,
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val_loss,
                    'best_val_ppl': math.exp(min(best_val_loss, 50.0)),
                    'final_val_loss': avg_val_loss,
                    'final_val_ppl': val_ppl,
                    'final_val_acc': val_acc,
                    'converged': bool(improvement < convergence_threshold),
                    'last_epoch_improvement': improvement,
                },
                stage="pretrain",
            )
            # A still-descending final epoch means the run stopped early: say so
            # in the log instead of letting "epoch 20" imply convergence.
            if improvement >= convergence_threshold:
                print(f"  NOTE: val loss still improving by {improvement:.4f} in the "
                      f"last epoch — this run is NOT converged; train longer or "
                      f"export best_model.pt.")

    # Final checkpoint
    if is_main_process:
        final_path = os.path.join(hp_config.checkpoint_dir, "final_model.pt")
        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save(model_to_save.state_dict(), final_path)
        print(f"\nTraining complete! Final model saved to: {final_path}")
        print(f"Best epoch {best_epoch}: val loss {best_val_loss:.4f} "
              f"(PPL {math.exp(min(best_val_loss, 50.0)):.3f})")
        print(f"Export with: python scripts/convert_to_hf.py --config-path={config_path} "
              f"--checkpoint={os.path.join(hp_config.checkpoint_dir, 'best_model.pt')}")
        writer.close()

        model_label = Path(hp_config.checkpoint_dir).resolve().parent.name
        for metric_name, metric_value in (
            ("best_val_loss", best_val_loss),
            ("best_val_ppl", math.exp(min(best_val_loss, 50.0))),
            ("final_val_loss", epoch_history[-1]["val_loss"] if epoch_history else float("nan")),
        ):
            record_result(stage="pretrain", model=model_label, metric=metric_name,
                          value=metric_value, split="corpus-test",
                          seed=hp_config.seed, config_hash=hp_config.config_hash())

    cleanup_distributed()
    
if __name__ == "__main__":
    main()
