# Aspect-Based Sentiment Analysis (ABSA) Implementation

## 1. Objective
This project implements **Aspect-Based Sentiment Analysis (ABSA)** using multiple deep learning models. The goal is to analyze sentiment for specific aspect terms in a given text.

## 2. Data Preprocessing
- Each aspect term is treated as a **separate instance**.
- A dictionary is created for each aspect term, containing:
  - **Tokens** (preprocessed text)
  - **Polarity** (sentiment label)
  - **Aspect Term** (target entity)
  - **Index** (position in text)

## 3. Model Training
Different deep learning models were trained using various pre-trained embeddings:

| Model                | Embedding Dimension | Hidden Layers | Dropout |
|----------------------|--------------------|--------------|---------|
| **RNN + GloVe**      | 50                 | 64           | 0.4     |
| **RNN + fastText**   | 300                | 32           | 0.4     |
| **RNN + BERT**       | 768                | 64           | 0.4     |
| **GRU + GloVe**      | 50                 | 64           | 0.4     |
| **GRU + BERT**       | 768                | 64           | 0.4     |
| **GRU + fastText**   | 300                | 32           | 0.4     |
| **LSTM + BERT**      | 768                | 64           | 0.4     |
| **LSTM + fastText**  | 300                | 32           | 0.4     |
| **LSTM + GloVe**     | 50                 | 64           | 0.4     |

## 4. Best Model Performance
The best-performing model was **LSTM + BERT**, trained for **20 epochs** with the following metrics:

### Training Metrics
- **Final Training Accuracy:** 83.21% (Epoch 20)
- **Final Training Loss:** 0.3299 (Epoch 20)

### Validation Metrics
- **Final Validation Accuracy:** 55.58% (Epoch 20)
- **Final Validation Loss:** 1.3295 (Epoch 20)

## 5. Additional Task
We fine-tuned **BERT, BART, and RoBERTa** on the preprocessed dataset and evaluated their performance.

| Model    | Validation Accuracy |
|----------|--------------------|
| **BERT** | 74.92%             |
| **BART** | 72.54%             |
| **RoBERTa** | 74.92%          |

## 6. Training and Validation Loss Plot
![download](https://github.com/user-attachments/assets/56d11f33-cb88-496d-9525-8c0d57e5f948)
![bart](https://github.com/user-attachments/assets/e4ed8cf3-5bd8-45d1-8bdd-9668c0e1af00)
![roberta](https://github.com/user-attachments/assets/37150465-04d6-4613-b911-3b4cc9bacc90)



## 8. Acknowledgments
- **GloVe, fastText, BERT, BART, and RoBERTa embeddings**.
- **Hugging Face Transformers library** for model fine-tuning.
- **Aspect-Based Sentiment Analysis techniques** for structured NLP insights.
