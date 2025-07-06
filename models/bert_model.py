from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import AdamWeightDecay
import tensorflow as tf
import numpy as np

def train_bert(train_texts, test_texts, train_labels, test_labels):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_enc = tokenizer(train_texts, truncation=True, padding=True, return_tensors="tf", max_length=128)
    test_enc = tokenizer(test_texts, truncation=True, padding=True, return_tensors="tf", max_length=128)

    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    
    # Use the recommended optimizer for Transformers
    optimizer = AdamWeightDecay(learning_rate=3e-5, weight_decay_rate=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    model.fit(
        dict(train_enc),
        np.array(train_labels),
        validation_split=0.1,
        epochs=3,
        batch_size=8,
        verbose=1
    )
    preds = model.predict(dict(test_enc)).logits
    return np.argmax(preds, axis=1)