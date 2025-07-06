import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

def train_gru(X_train, y_train, X_test, vocab_size):
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
    model = Sequential([
        Embedding(vocab_size, 128, input_length=X_train.shape[1]),
        GRU(64),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train_cat, epochs=5, batch_size=32, verbose=0)
    preds = model.predict(X_test)
    return np.argmax(preds, axis=1)
