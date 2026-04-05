import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import config

logger = logging.getLogger(__name__)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

class TransformerEngine:
    def __init__(self):
        self.model_name = config.TRANSFORMER_MODEL_NAME
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.to(self.device)

    def train(self, X_train, y_train):
        logger.info(f"Starting DistilBERT fine-tuning on {self.device}...")
        
        train_dataset = ReviewDataset(X_train if isinstance(X_train, list) else X_train.tolist(), 
                                      y_train if isinstance(y_train, list) else y_train.tolist(), 
                                      self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=config.TRANSFORMER_BATCH_SIZE, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.model.train()
        
        for epoch in range(config.TRANSFORMER_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{config.TRANSFORMER_EPOCHS}")
            total_loss = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_loss:.4f}")
            
        logger.info("Training completed.")
        self.save_model()  # Save immediately after training

    def predict(self, texts):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch_texts = list(texts[i:i + 32])
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                encodings = {k: v.to(self.device) for k, v in encodings.items()}
                outputs = self.model(**encodings)
                preds = outputs.logits.argmax(dim=1).tolist()
                all_preds.extend(preds)

        return all_preds

    def save_model(self):
        save_path = config.TRANSFORMER_SAVE_PATH
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Transformer model and tokenizer saved at: {save_path}")
        return save_path

    def load_model(self, path=config.TRANSFORMER_SAVE_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Loaded Transformer model from {path}")
