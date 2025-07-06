from preprocessing import load_and_preprocess_data
from utils import print_and_store_results, plot_results

from models.spacy_model import train_spacy
from models.gru_model import train_gru
from models.transformer_model import train_transformer
from models.bert_model import train_bert

print("🔎 Loading and preprocessing data...")
train_texts, test_texts, X_train, X_test, y_train, y_test, tokenizer, le = load_and_preprocess_data("data.csv")

vocab_size = len(tokenizer.word_index) + 1
results = {}

# spaCy
spacy_preds = train_spacy(train_texts, y_train, test_texts, le)
print_and_store_results("spaCy", y_test, spacy_preds, results)

# GRU
gru_preds = train_gru(X_train, y_train, X_test, vocab_size)
print_and_store_results("GRU", y_test, gru_preds, results)

# Transformer
transformer_preds = train_transformer(X_train, y_train, X_test, vocab_size)
print_and_store_results("Transformer", y_test, transformer_preds, results)

# BERT
bert_preds = train_bert(train_texts, test_texts, y_train, y_test)
print_and_store_results("BERT", y_test, bert_preds, results)

plot_results(results)
