import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer, BartForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, get_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load dataset
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Custom dataset for PyTorch
class ABSADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2, "conflict":2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        tokens = instance["tokens"]
        aspect = instance["aspect_term"]
        polarity = instance["polarity"]

        # Convert tokens to sentence
        sentence = " ".join(tokens)
        inputs = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Get labels
        label = torch.tensor(self.label_map[polarity], dtype=torch.long)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label
        }

# Function to train a model
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-5, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_acc = correct / total

        model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        val_losses.append(total_loss / len(val_loader))
        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_losses[-1]:.4f}, Val Acc={val_acc:.4f}")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses, best_model_state

# Updated plot_loss function
def plot_loss(train_losses, val_losses, title):
    plt.plot(train_losses, label="Train Loss", linestyle='--')
    plt.plot(val_losses, label="Validation Loss", linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    train_data = load_data("train_task_2.json")
    val_data = load_data("val_task_2.json")

    # Initialize tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Create datasets
    train_dataset_bert = ABSADataset(train_data, bert_tokenizer)
    val_dataset_bert = ABSADataset(val_data, bert_tokenizer)

    train_dataset_bart = ABSADataset(train_data, bart_tokenizer)
    val_dataset_bart = ABSADataset(val_data, bart_tokenizer)

    train_dataset_roberta = ABSADataset(train_data, roberta_tokenizer)
    val_dataset_roberta = ABSADataset(val_data, roberta_tokenizer)

    # Create DataLoaders
    train_loader_bert = DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
    val_loader_bert = DataLoader(val_dataset_bert, batch_size=16, shuffle=False)

    train_loader_bart = DataLoader(train_dataset_bart, batch_size=16, shuffle=True)
    val_loader_bart = DataLoader(val_dataset_bart, batch_size=16, shuffle=False)

    train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=16, shuffle=True)
    val_loader_roberta = DataLoader(val_dataset_roberta, batch_size=16, shuffle=False)

    print("Training BERT model...")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    train_losses_bert, val_losses_bert, best_bert_state = train_model(bert_model, train_loader_bert, val_loader_bert)
    torch.save(best_bert_state, "best_bert_model.pth")

    print("Training BART model...")
    bart_model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=3)
    train_losses_bart, val_losses_bart, best_bart_state = train_model(bart_model, train_loader_bart, val_loader_bart)
    torch.save(best_bart_state, "best_bart_model.pth")

    print("Training RoBERTa model...")
    roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
    train_losses_roberta, val_losses_roberta, best_roberta_state = train_model(roberta_model, train_loader_roberta, val_loader_roberta)
    torch.save(best_roberta_state, "best_roberta_model.pth")

    print("Plotting losses...")
    plot_loss(train_losses_bert, val_losses_bert, "BERT Training and Validation Loss")
    plot_loss(train_losses_bart, val_losses_bart, "BART Training and Validation Loss")
    plot_loss(train_losses_roberta, val_losses_roberta, "RoBERTa Training and Validation Loss")

    print("Training complete. Models saved.")