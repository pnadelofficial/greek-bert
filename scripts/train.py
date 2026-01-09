from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import os
import argparse
from utils import TrainingConfig, setup_distributed, cleanup_distributed, mlm_masking

parser = argparse.ArgumentParser(description="Reads in training YAML file.")
parser.add_argument(
    "--config-path",
    dest="config_path",
)
args = parser.parse_args()
config_path = args.config_path

def main():
    # init and check GPUs
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')

    np_rng = np.random.default_rng(seed=22091997 + rank) # diff seed for each process

    if is_main_process:
        print(f"Training on {world_size} GPUs with mixed precision")
        print(f"Local rank: {local_rank}, Global rank: {rank}")
    
    # read in config
    hp_config = TrainingConfig.from_yaml(config_path)
    
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
        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
        model = AutoModelForMaskedLM.from_config(config).to(device)
        tokenizer = AutoTokenizer.from_pretrained("modernbert-greek-tokenizer")
        vocab_size = len(tokenizer.get_vocab())
        config.vocab_size = vocab_size
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
    else:
        config = AutoConfig.from_pretrained(hp_config.pretrained_model)
        model = AutoModelForMaskedLM.from_pretrained(hp_config.pretrained_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hp_config.pretrained_model)
        vocab_size = len(tokenizer.get_vocab())
    
    tokenizer.eos_token = tokenizer.pad_token

    # tokenization - TODO should probably do this ahead of time
    if is_main_process:
        print("Tokenizing dataset...")

    # wrap with ddp
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # masking
    mask_token = tokenizer.mask_token_id
    pad_token = tokenizer.pad_token_id
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
        num_workers=4,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        tokenized_datasets["test"], 
        batch_size=hp_config.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(hp_config.lr),
        weight_decay=float(hp_config.weight_decay)
    )
    
    # amp scaler/scheduler
    scaler = GradScaler() if hp_config.use_mixed_precision else None
    
    if is_main_process:
        print(f"DEBUG: num_epochs = {hp_config.num_epochs}, type = {type(hp_config.num_epochs)}")
        print(f"DEBUG: len(train_dataloader) = {len(train_dataloader)}, type = {type(len(train_dataloader))}")
        print(f"DEBUG: gradient_accumulation_steps = {hp_config.gradient_accumulation_steps}, type = {type(hp_config.gradient_accumulation_steps)}")

        num_training_steps = hp_config.num_epochs * len(train_dataloader) // hp_config.gradient_accumulation_steps
        print(f"DEBUG: num_training_steps = {num_training_steps}, type = {type(num_training_steps)}")
        print(f"DEBUG: lr = {hp_config.lr}, type = {type(hp_config.lr)}")
        print(f"DEBUG: max_lr = {hp_config.max_lr}, type = {type(hp_config.max_lr)}")
        print(f"DEBUG: pct_start = {hp_config.pct_start}, type = {type(hp_config.pct_start)}")
        print(f"DEBUG: weight_decay = {hp_config.weight_decay}, type = {type(hp_config.weight_decay)}")
    
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
        print(f"Mixed precision: {hp_config.use_mixed_precision}")
        print(f"Gradient accumulation steps: {hp_config.gradient_accumulation_steps}")
    
    global_step = 0
    model.train()
    
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

            masked_input_ids, masked_labels = mlm_masking(input_ids, np_rng, hp_config.mask_prob, mask_token, pad_token, ignore_index, vocab_size)
            masked_input_ids = masked_input_ids.to(device)
            masked_labels = masked_labels.to(device)

            # amp forward pass
            if hp_config.use_mixed_precision:
                with autocast():
                    outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=masked_labels)
                    loss = outputs.loss / hp_config.gradient_accumulation_steps
            else:
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=masked_labels)
                loss = outputs.loss / hp_config.gradient_accumulation_steps

            epoch_loss += loss.item() * hp_config.gradient_accumulation_steps
            num_batches += 1
            
            # amp backward pass
            if hp_config.use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # update after grad accumulation
            if (step + 1) % hp_config.gradient_accumulation_steps == 0:
                if hp_config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # logging
                if is_main_process and global_step % hp_config.log_interval == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    writer.add_scalar('Loss/train', loss.item() * hp_config.gradient_accumulation_steps, global_step)
                    writer.add_scalar('Learning_rate', current_lr, global_step)
                    
                    iterator.set_postfix({
                        'loss': f'{loss.item() * hp_config.gradient_accumulation_steps:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })

                global_step += 1

        # end of epoch
        avg_epoch_loss = epoch_loss / num_batches
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
                    'scaler_state_dict': scaler.state_dict() if hp_config.use_mixed_precision else None,
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
        with torch.no_grad():
            val_iterator = tqdm(valid_dataloader, disable=not is_main_process, desc="Validation")
            for batch in val_iterator:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                masked_input_ids, masked_labels = mlm_masking(input_ids, np_rng, hp_config.mask_prob, mask_token, pad_token, ignore_index, vocab_size)
                masked_input_ids = masked_input_ids.to(device)
                masked_labels = masked_labels.to(device)

                if hp_config.use_mixed_precision:
                    with autocast():
                        outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=masked_labels)
                else:
                    outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=masked_labels)
                
                val_loss += outputs.loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        
        if is_main_process:
            print(f"Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        model.train()

    # Final checkpoint
    if is_main_process:
        final_path = os.path.join(hp_config.checkpoint_dir, "final_model.pt")
        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save(model_to_save.state_dict(), final_path)
        print(f"\nTraining complete! Final model saved to: {final_path}")
        writer.close()

    cleanup_distributed()
    
if __name__ == "__main__":
    main()
