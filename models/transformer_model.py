import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Dropout, Dense
import numpy as np

class SimpleTransformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = LayerNormalization()
        self.pool = GlobalAveragePooling1D()
        self.dropout = Dropout(0.1)
        self.fc = Dense(3, activation="softmax")

    def call(self, x):
        x = self.embedding(x)
        attn_output = self.mha(x, x)
        x = self.norm(x + attn_output)
        x = self.pool(x)
        x = self.dropout(x)
        return self.fc(x)

def train_transformer(X_train, y_train, X_test, vocab_size):
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
    model = SimpleTransformer(vocab_size)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train_cat, epochs=5, batch_size=32, verbose=0)
    preds = model.predict(X_test)
    return np.argmax(preds, axis=1)
