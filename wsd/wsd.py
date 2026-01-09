from pathlib import Path
import json
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

# glaux data read in
glaux_dir = Path("/cluster/tufts/tuftsai/pnadel01/greek-bert/wsd/glaux/xml")
## glaux sentences
with open(glaux_dir / 'glaux_sentences.json', 'r', encoding='utf-8') as f:
    sentences = json.load(f)
## glaux ids to word ids
with open(glaux_dir / 'glaux_id2word_id.json', 'r', encoding='utf-8') as f:
    glaux_id2word_id = json.load(f)

# config
@dataclass
class WSDConfig:
    model_name: str = "/cluster/tufts/tuftsai/pnadel01/greek-bert/hf_format" # defaults to our model
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    dropout: float = 0.1
    test_size: float = 0.2
    random_state: int = 42

# dataset
class WSDDataset(Dataset):
    def __init__(self, data_path, glaux_data, tokenizer, target_word, max_length=512):
        self.tokenizer = tokenizer
        self.target_word = target_word
        self.max_length = max_length

        df = pd.read_csv(data_path, sep="\t", header=None)

        if len(df.columns) == 2:
            df.columns = ['glaux_id', 'sense']
        else:
            df.columns = ['glaux_id', 'sense', 'subsense']

        self.data = []
        missing_ids = []

        for idx, row in df.iterrows():
            word_id = str(row["glaux_id"])
            glaux_id = glaux_id2word_id.get(word_id, None)
            sense = row["sense"]

            if glaux_id in glaux_data:
                sentense = glaux_data[glaux_id]
                self.data.append({
                    'glaux_id': glaux_id,
                    'text': sentense['text'],
                    'sense': sense,
                    'target_position': sentense['word_ids'].index(word_id)
                })
            else:
                missing_ids.append(glaux_id)
        
        if missing_ids:
            print(f"Warning: {len(missing_ids)} glaux_ids not found in glaux data.")
        print(f"Loaded {len(self.data)} examples for target word '{self.target_word}'.")

        self.sense_to_id = {sense: idx for idx, sense in enumerate(sorted(set(df['sense'])))}
        self.id_to_sense = {idx: sense for sense, idx in self.sense_to_id.items()}
        self.num_senses = len(self.sense_to_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['text']
        sense_label = self.sense_to_id[item['sense']]
        target_position = item['target_position']

        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target_position': target_position,
            'label': torch.tensor(sense_label, dtype=torch.long),
            'glaux_id': item['glaux_id']
        }

# model
class WSDClassifier(nn.Module):    
    def __init__(self, bert_model, num_senses: int, dropout: float = 0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_senses)
        
    def forward(self, input_ids, attention_mask, target_position):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        batch_size = sequence_output.size(0)
        
        target_embeddings = sequence_output[
            torch.arange(batch_size, device=sequence_output.device),
            target_position
        ]  # [batch_size, hidden_size]
        
        target_embeddings = self.dropout(target_embeddings)
        logits = self.classifier(target_embeddings)
        
        return logits

# trainer
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            target_positions = batch['target_position'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, target_positions)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Evaluating')
            for batch in pbar:
                input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
                attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
                target_positions = batch['target_position'].to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

                logits = self.model(input_ids, attention_mask, target_positions)
                loss = self.criterion(logits, labels)

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy, all_preds, all_labels

    def train(self, save_dir="./wsd_model"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

            val_loss, val_acc, val_preds, val_labels = self.evaluate()
            print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, save_path / 'best_model.pt')
                print(f"Saved best model with Val Acc={val_acc:.4f} at epoch {epoch}.")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, save_path / 'final_model.pt')

        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print("Training complete. Best Val Acc: {:.4f}".format(self.best_val_acc))
        return self.history

def main():
    wsd_base_path = Path("/cluster/tufts/tuftsai/pnadel01/greek-bert/wsd/ancient-greek-wsd-data")

    # our model
    print("Loading our model and tokenizer...")
    config = WSDConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

    # harmonia classifier
    print("Preparing dataset for 'ἁρμονιά'...")
    har_ds = WSDDataset(
        data_path=wsd_base_path / 'harmonia_glaux.txt',
        tokenizer=tokenizer,
        glaux_data=sentences,
        target_word='ἁρμονιά',
        max_length=config.max_length
    )

    train_size = int((1 - config.test_size) * len(har_ds))
    val_size = len(har_ds) - train_size

    har_train_dataset, har_val_dataset = torch.utils.data.random_split(
        har_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_state)
    )
    har_train_loader = DataLoader(
        har_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0 
    )
    har_val_loader = DataLoader(
        har_val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    har_classifier = WSDClassifier(bert_model=model, num_senses=har_ds.num_senses, dropout=config.dropout)
    har_classifier = har_classifier.to('cuda' if torch.cuda.is_available() else 'cpu')

    har_trainer = Trainer(
        model=har_classifier,
        train_loader=har_train_loader,
        val_loader=har_val_loader,
        config=config
    )
    har_trainer.train(save_dir="./our_wsd_model/harmonia")

    # kosmos classifier
    print("Preparing dataset for 'κόσμος'...")
    kos_ds = WSDDataset(
        data_path=wsd_base_path / 'kosmos_glaux.txt',
        tokenizer=tokenizer,
        glaux_data=sentences,
        target_word='κόσμος',
        max_length=config.max_length
    ) 

    train_size = int((1 - config.test_size) * len(kos_ds))
    val_size = len(kos_ds) - train_size

    kos_train_dataset, kos_val_dataset = torch.utils.data.random_split(
        kos_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_state)
    )
    kos_train_loader = DataLoader(
        kos_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0 
    )
    kos_val_loader = DataLoader(
        kos_val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    ) 
    kos_classifier = WSDClassifier(bert_model=model, num_senses=kos_ds.num_senses, dropout=config.dropout)
    kos_classifier = kos_classifier.to('cuda' if torch.cuda.is_available() else 'cpu') 

    kos_trainer = Trainer(
        model=kos_classifier,
        train_loader=kos_train_loader,
        val_loader=kos_val_loader,
        config=config
    )
    kos_trainer.train(save_dir="./our_wsd_model/kosmos")

    # jacobo model
    print("Loading Jacobo model and tokenizer...")
    ari_config = WSDConfig(model_name="Jacobo/aristoBERTo")
    ari_tokenizer = AutoTokenizer.from_pretrained(ari_config.model_name)
    ari_model = AutoModel.from_pretrained(ari_config.model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

    print("Preparing dataset for 'ἁρμονιά' with Jacobo model...")
    har_ds_ari = WSDDataset(
        data_path=wsd_base_path / 'harmonia_glaux.txt',
        tokenizer=ari_tokenizer,
        glaux_data=sentences,
        target_word='ἁρμονιά',
        max_length=ari_config.max_length
    )

    train_size = int((1 - ari_config.test_size) * len(har_ds_ari))
    val_size = len(har_ds_ari) - train_size

    har_train_dataset_ari, har_val_dataset_ari = torch.utils.data.random_split(
        har_ds_ari,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(ari_config.random_state)
    )

    har_train_loader_ari = DataLoader(
        har_train_dataset_ari,
        batch_size=ari_config.batch_size,
        shuffle=True,
        num_workers=0 
    )
    har_val_loader_ari = DataLoader(
        har_val_dataset_ari,
        batch_size=ari_config.batch_size,
        shuffle=False,
        num_workers=0
    )  

    har_classifier_ari = WSDClassifier(bert_model=ari_model, num_senses=har_ds_ari.num_senses, dropout=ari_config.dropout)
    har_classifier_ari = har_classifier_ari.to('cuda' if torch.cuda.is_available() else 'cpu') 

    har_trainer_ari = Trainer(
        model=har_classifier_ari,
        train_loader=har_train_loader_ari,
        val_loader=har_val_loader_ari,
        config=ari_config
    )
    har_trainer_ari.train(save_dir="./jacobo_wsd_model/harmonia")   

    print("Preparing dataset for 'κόσμος' with Jacobo model...")
    kos_ds_ari = WSDDataset(
        data_path=wsd_base_path / 'kosmos_glaux.txt',
        tokenizer=ari_tokenizer,
        glaux_data=sentences,
        target_word='κόσμος',
        max_length=ari_config.max_length
    )

    train_size = int((1 - ari_config.test_size) * len(kos_ds_ari))
    val_size = len(kos_ds_ari) - train_size

    kos_train_dataset_ari, kos_val_dataset_ari = torch.utils.data.random_split(
        kos_ds_ari,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(ari_config.random_state)
    )

    kos_train_loader_ari = DataLoader(
        kos_train_dataset_ari,
        batch_size=ari_config.batch_size,
        shuffle=True,
        num_workers=0
    )
    kos_val_loader_ari = DataLoader(
        kos_val_dataset_ari,
        batch_size=ari_config.batch_size,
        shuffle=False,
        num_workers=0
    )

    kos_classifier_ari = WSDClassifier(bert_model=ari_model, num_senses=kos_ds_ari.num_senses, dropout=ari_config.dropout)
    kos_classifier_ari = kos_classifier_ari.to('cuda' if torch.cuda.is_available() else 'cpu')

    kos_trainer_ari = Trainer(
        model=kos_classifier_ari,
        train_loader=kos_train_loader_ari,
        val_loader=kos_val_loader_ari,
        config=ari_config 
    )
    kos_trainer_ari.train(save_dir="./jacobo_wsd_model/kosmos")

if __name__ == "__main__":
    main()