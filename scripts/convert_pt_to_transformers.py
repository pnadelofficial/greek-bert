from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
import torch
import argparse
from utils import TrainingConfig

parser = argparse.ArgumentParser(description="Reads in training YAML file.")
parser.add_argument(
    "--config-path",
    dest="config_path",
)
args = parser.parse_args()
config_path = args.config_path

def main():
    device = torch.device(f'cuda:0')
    hp_config = TrainingConfig.from_yaml(config_path)
    if not hp_config.pretrained_model:
        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
        model = AutoModelForMaskedLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained("modernbert-greek-tokenizer")
    else:
        config = AutoConfig.from_pretrained(hp_config.pretrained_model)
        model = AutoModelForMaskedLM.from_pretrained(hp_config.pretrained_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hp_config.pretrained_model)
        vocab_size = len(tokenizer.get_vocab())
        
    tokenizer.eos_token = tokenizer.pad_token

    checkpoint = torch.load("/cluster/tufts/tuftsai/pnadel01/greek-bert/scripts/checkpoints/final_model.pt")
    state_dict = {k: v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    
    output_dir = "/cluster/tufts/tuftsai/pnadel01/greek-bert/hf_format"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()