# Financial Sentiment Analysis

This project compares four NLP models on a financial sentiment dataset using the:

- spaCy TextClassifier
- GRU (TensorFlow)
- Transformer (TensorFlow)
- BERT (Hugging Face)
  The Dataset is taken from Kaggle.
## Structure

- `main.py`: Entry point, trains and compares all models
- `models/`: Contains implementation of each model
- `preprocessing.py`: Cleans and tokenizes text
- `utils.py`: Evaluation and plotting
- `financial_sentiment.csv`: Input dataset (3 sentiment classes)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Output

- Console shows accuracy for each model
- `model_comparison.png` shows a bar chart
