from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
import torch

config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModelForMaskedLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("modernbert-greek-tokenizer")

checkpoint = torch.load("/cluster/tufts/tuftsai/pnadel01/greek_bert/scripts/checkpoints/final_model.pt")
state_dict = {k: v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)

output_dir = "/cluster/tufts/tuftsai/pnadel01/greek_bert/hf_format"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


