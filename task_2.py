import json
import re
import torch
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import shutil  # For copying best overall model

# Load data from file
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Preprocess the data (tokenization and aspect terms extraction)
def preprocess(input_file, output_file):
    data = load_data(input_file)
    processed_data = []
    for instance in data:
        tokens = re.findall(r"\w+|[^\w\s]", instance["sentence"])
        aspect_terms = instance["aspect_terms"]
        for aspect in aspect_terms:
            term = aspect["term"]
            polarity = aspect["polarity"]
            index = tokens.index(term) if term in tokens else -1
            if index != -1:
                processed_data.append({
                    "tokens": tokens,
                    "polarity": polarity,
                    "aspect_term": term,
                    "index": index,
                    "sentence": instance["sentence"] 
                })
    with open(output_file, "w") as f:
        json.dump(processed_data, f)
        f.write("\n\n")  # Adds a newline gap between entries
    print(f"Preprocessed data saved to {output_file}")

# Build vocabulary from dataset
def build_vocabulary(file):
    vocab = set()
    for instance in file:
        vocab.update(instance["tokens"])
    vocab_list = ["<unk>", "<pad>"] + sorted(vocab)
    vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}  # Convert to dictionary
    return vocab_dict

# Encode sentence by replacing words with their corresponding vocab indices
def encode_sentence(sentence, vocab):
    return [vocab[word] if word in vocab else vocab["<unk>"] for word in sentence]

# Prepare data loader for training and validation data
def prepare_data(dataset, vocab, use_bert=False):
    sentences = []
    labels = []
    SENTIMENT_LABELS = {"negative": 0, "neutral": 1, "positive": 2,"conflict":3}  
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if use_bert else None
    
    for instance in dataset:
        if use_bert:
            # Tokenize the sentence using BERT tokenizer
            inputs = tokenizer(instance["sentence"], return_tensors="pt", padding=True, truncation=True)
            encoded_sentence = inputs["input_ids"].squeeze(0)  
        else:
            encoded_sentence = torch.tensor(encode_sentence(instance["tokens"], vocab))
        
        # Only assign a label if it's one of the valid classes
        if instance["polarity"] in SENTIMENT_LABELS:
            sentence_labels = [SENTIMENT_LABELS[instance["polarity"]]] * len(encoded_sentence)
            sentences.append(encoded_sentence)
            labels.append(torch.tensor(sentence_labels))  # Assigning label to each token in the sentence
    
    sentences_padded = rnn.pad_sequence(sentences, batch_first=True, padding_value=vocab["<pad>"])
    labels_padded = rnn.pad_sequence(labels, batch_first=True, padding_value=-1)  # Padding labels with -1 (ignore_index for loss)
    
    return DataLoader(TensorDataset(sentences_padded, labels_padded), batch_size=32, shuffle=True)

# Save embeddings to pickle file for faster access
def save_embeddings_pickle(embeddings, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {file_path}")

# Load embeddings either from cache or directly from a text file
def load_embeddings(file_path, embedding_dim=100, pickle_path=None):
    if pickle_path and os.path.exists(pickle_path):
        print(f"Loading embeddings from cache: {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    
    print(f"Loading embeddings from text file: {file_path} (this may take time)...")
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float64)
            embeddings[word] = vector
    
    if pickle_path:
        save_embeddings_pickle(embeddings, pickle_path)
    
    return embeddings

# Define model class for aspect term extraction using RNN, GRU, or LSTM
class AspectTermExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None, model_type="RNN", dropout_rate=0.4, use_bert=False):
        super(AspectTermExtractor, self).__init__()
        self.use_bert = use_bert

        if self.use_bert:
            # Initialize BERT model
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        else:
            # Use GloVe or fastText embeddings
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            self.hidden_dim = hidden_dim

            if model_type == "RNN":
                self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            elif model_type == "GRU":
                self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            elif model_type == "LSTM":
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.use_bert:
            # Get BERT outputs
            outputs = self.bert(x)
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
            out = self.fc(last_hidden_state)
        else:
            # Use GloVe, fastText, or LSTM
            embedded = self.embedding(x)
            rnn_out, _ = self.rnn(embedded)
            out = self.dropout(rnn_out)
            out = self.fc(out)
        
        return F.log_softmax(out, dim=2)

# Train the model with training and validation data
def train_model(model, train_loader, val_loader, epochs=20, lr=0.00001, model_name="best_model.pth"):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # ignore <pad> tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []  # List to store training accuracies
    val_accuracies = []  # List to store validation accuracies

    best_val_accuracy = 0  # Track best validation accuracy
    best_model_state_dict = None  # To save the best model

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions_train = 0
        total_tokens_train = 0

        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(sentences)  # Shape: (batch_size, seq_len, output_dim)

            # Flatten outputs to shape (batch_size * seq_len, output_dim)
            outputs = outputs.view(-1, outputs.shape[2])
            
            # Flatten labels to shape (batch_size * seq_len, )
            labels = labels.view(-1)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            predicted_labels = outputs.argmax(dim=1)
            correct_predictions_train += (predicted_labels == labels).sum().item()
            total_tokens_train += (labels != -1).sum().item()

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(correct_predictions_train / total_tokens_train)

        model.eval()
        val_loss = 0
        correct_predictions_val = 0
        total_tokens_val = 0
        with torch.no_grad():
            for sentences, labels in val_loader:
                sentences, labels = sentences.to(device), labels.to(device)
                outputs = model(sentences)

                outputs = outputs.view(-1, outputs.shape[2])
                labels = labels.view(-1)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                predicted_labels = outputs.argmax(dim=1)
                correct_predictions_val += (predicted_labels == labels).sum().item()
                total_tokens_val += (labels != -1).sum().item()

        val_losses.append(val_loss / len(val_loader))
        current_val_accuracy = correct_predictions_val / total_tokens_val
        val_accuracies.append(current_val_accuracy)

        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {current_val_accuracy:.4f}")

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_model_state_dict = model.state_dict()

    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, model_name)
        print(f"Best model saved to {model_name} with validation accuracy: {best_val_accuracy:.4f}")
    return train_losses, val_losses, train_accuracies, val_accuracies

# Plot loss over training epochs
def plot_losses(train_losses, val_losses, title):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.show()

# Load the trained model from the saved file
def load_trained_model(model_class, model_path, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None, use_bert=False):
    model = model_class(vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings, use_bert=use_bert)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def get_test_accuracy(model_class, model_path, test_file, output_file, vocab, use_bert=False):
    # Preprocess the test data using the existing preprocess function
    preprocess_test_data(test_file, output_file)
    
    # Load the processed test data
    test_data = load_data(output_file)  # Assuming this loads the processed data into the format we need
    
    # Prepare the test DataLoader (similar to training data)
    test_loader = prepare_data(test_data, vocab, use_bert)
    
    # Load the trained model (best saved model)
    model = load_trained_model(model_class, model_path, len(vocab), 50, 64, 3, glove_tensor, use_bert)
    
    # Evaluate the model on the test data
    accuracy = evaluate_model_on_test(model, test_loader)
    return accuracy

# Process the test data using the existing preprocess function
def preprocess_test_data(input_file, output_file):
    preprocess(input_file, output_file)

# Evaluate the model on the test data
def evaluate_model_on_test(model, test_loader):
    correct_predictions = 0
    total_tokens = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for sentences, labels in test_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences)
            outputs = outputs.view(-1, outputs.shape[2])
            labels = labels.view(-1)
            
            predicted_labels = outputs.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_tokens += (labels != -1).sum().item()

    accuracy = correct_predictions / total_tokens
    return accuracy

# Main function
if __name__ == "__main__":
    train_file = "train.json"
    val_file = "val.json"
    train_output = "train_task_2.json"
    val_output = "val_task_2.json"

    # Uncomment if you need to preprocess the raw train/val data
    # preprocess(train_file, train_output)
    # preprocess(val_file, val_output)

    print("Loading the data...")
    train_data = load_data("train_task_2.json")
    val_data = load_data("val_task_2.json")

    vocab = build_vocabulary(train_data)

    glove_path = "glove.6B.50d.txt"
    fasttext_path = "crawl-300d-2M.vec"

    glove_pickle_path = "glove_50d.pkl"
    fastText_pickle_path = "fastText_300d.pkl"

    glove_dim = 50
    fastText_dim = 300

    glove_embeddings = load_embeddings(glove_path, glove_dim, glove_pickle_path)
    fasttext_embeddings = load_embeddings(fasttext_path, fastText_dim, fastText_pickle_path)

    train_loader = prepare_data(train_data, vocab)
    val_loader = prepare_data(val_data, vocab)

    vocab_list = list(vocab.keys())

    # Convert GloVe embeddings to tensor
    glove_matrix = np.array([glove_embeddings.get(w, np.zeros(glove_dim, dtype=np.float32)) for w in vocab_list])
    glove_tensor = torch.tensor(glove_matrix, dtype=torch.float32)

    # Convert fastText embeddings to tensor
    fasttext_matrix = np.array([fasttext_embeddings.get(w, np.zeros(fastText_dim, dtype=np.float32)) for w in vocab_list])
    fasttext_tensor = torch.tensor(fasttext_matrix, dtype=torch.float32)

    # Initialize tracking variables for the best overall model
    best_overall_accuracy = 0
    best_overall_model_path = None

    # # Train RNN + GloVe
    # print("Training RNN + GloVe...")
    # rnn_glove = AspectTermExtractor(len(vocab), 50, 64, 3, glove_tensor, model_type="RNN")
    # losses, _, _, val_acc = train_model(rnn_glove, train_loader, val_loader, model_name="rnn_glove.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "rnn_glove.pth"

    # # Train RNN + fastText
    # print("Training RNN + fastText...")
    # rnn_fasttext = AspectTermExtractor(len(vocab), 300, 32, 3, fasttext_tensor, model_type="RNN")
    # losses, _, _, val_acc = train_model(rnn_fasttext, train_loader, val_loader, model_name="rnn_fasttext.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "rnn_fasttext.pth"

    # # Train GRU + GloVe
    # print("Training GRU + GloVe...")
    # gru_glove = AspectTermExtractor(len(vocab), 50, 64, 3, glove_tensor, model_type="GRU")
    # losses, _, _, val_acc = train_model(gru_glove, train_loader, val_loader, model_name="gru_glove.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "gru_glove.pth"

    # # Train GRU + fastText
    # print("Training GRU + fastText...")
    # gru_fasttext = AspectTermExtractor(len(vocab), 300, 32, 3, fasttext_tensor, model_type="GRU")
    # losses, _, _, val_acc = train_model(gru_fasttext, train_loader, val_loader, model_name="gru_fasttext.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "gru_fasttext.pth"

    # # Train LSTM + GloVe
    # print("Training LSTM + GloVe...")
    # lstm_glove = AspectTermExtractor(len(vocab), 50, 64, 3, glove_tensor, model_type="LSTM")
    # losses, _, _, val_acc = train_model(lstm_glove, train_loader, val_loader, model_name="lstm_glove.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "lstm_glove.pth"

    # # Train LSTM + fastText
    # print("Training LSTM + fastText...")
    # lstm_fasttext = AspectTermExtractor(len(vocab), 300, 32, 3, fasttext_tensor, model_type="LSTM")
    # losses, _, _, val_acc = train_model(lstm_fasttext, train_loader, val_loader, model_name="lstm_fasttext.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "lstm_fasttext.pth"

    # # Train RNN + BERT
    # print("Training RNN + BERT...")
    # rnn_bert = AspectTermExtractor(len(vocab), 768, 64, 3, None, model_type="RNN", use_bert=True)
    # losses, _, _, val_acc = train_model(rnn_bert, train_loader, val_loader, model_name="rnn_bert.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "rnn_bert.pth"

    # # Train GRU + BERT
    # print("Training GRU + BERT...")
    # gru_bert = AspectTermExtractor(len(vocab), 768, 64, 3, None, model_type="GRU", use_bert=True)
    # losses, _, _, val_acc = train_model(gru_bert, train_loader, val_loader, model_name="gru_bert.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "gru_bert.pth"

    # # Train LSTM + BERT
    # print("Training LSTM + BERT...")
    # lstm_bert = AspectTermExtractor(len(vocab), 768, 64, 3, None, model_type="LSTM", use_bert=True)
    # losses, _, _, val_acc = train_model(lstm_bert, train_loader, val_loader, model_name="lstm_bert.pth")
    # current_best = max(val_acc)
    # if current_best > best_overall_accuracy:
    #     best_overall_accuracy = current_best
    #     best_overall_model_path = "lstm_bert.pth"

    # print(f"\nBest overall model: {best_overall_model_path} with validation accuracy: {best_overall_accuracy:.4f}")

    # # Optionally, copy the best overall model to a final file name
    # if best_overall_model_path:
    #     shutil.copy(best_overall_model_path, "best_overall_model.pth")
    #     print("Best overall model copied to best_overall_model.pth")

    # # Plot training losses for each configuration (example for one configuration)
    # print("Plotting training losses for RNN + GloVe...")
    # plot_losses(losses, val_losses, "RNN + GloVe Loss")

    # Uncomment and adjust below for testing if needed:
    test_file = "val.json"
    output_file = "test_task_2.json"
    model_path = "./lstm_bert.pth" 
    accuracy = get_test_accuracy(AspectTermExtractor, model_path, test_file, output_file, vocab, use_bert=False)
    print(f"Test Accuracy: {accuracy:.4f}")
