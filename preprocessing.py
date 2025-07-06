import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(path, maxlen=100, vocab_size=10000):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df = df[df["Sentiment"].isin(["positive", "neutral", "negative"])]
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Sentiment"])

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["Sentence"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    X_train = tokenizer.texts_to_sequences(train_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    return train_texts, test_texts, X_train, X_test, train_labels, test_labels, tokenizer, le
