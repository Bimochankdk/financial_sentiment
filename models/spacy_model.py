import spacy
import random
from spacy.training import Example

def train_spacy(train_texts, train_labels, test_texts, le):
    print("Training spaCy...")
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat", last=True)
    for label in le.classes_:
        textcat.add_label(label)

    train_data = list(zip(train_texts, [{"cats": {c: c == le.inverse_transform([y])[0] for c in le.classes_}} for y in train_labels]))
    optimizer = nlp.initialize()
    
    for i in range(5):
        random.shuffle(train_data)
        losses = {}
        batches = spacy.util.minibatch(train_data, size=8)
        for batch in batches:
            examples = []
            for text, annot in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annot)
                examples.append(example)
            nlp.update(examples, sgd=optimizer, losses=losses)

    predictions = [max(nlp(t).cats, key=nlp(t).cats.get) for t in test_texts]
    return le.transform(predictions)