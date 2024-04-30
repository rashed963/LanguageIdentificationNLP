import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data_sample(filepath, sample_size=1000, random_state=42):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.seed(random_state)
    sampled_indices = random.sample(range(len(lines)), sample_size)
    sampled_lines = [lines[i].strip() for i in sampled_indices]
    return sampled_lines

def train_model(X_train, y_train):
    model = make_pipeline(CountVectorizer(analyzer='char', ngram_range=(3, 3)), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model

def predict_language(text, model):
    return model.predict([text])[0]

if __name__ == "__main__":
    # Load data
    X_train = load_data_sample('data/train/x_train.txt')
    y_train = load_data_sample('data/train/y_train.txt')
    X_test = load_data_sample('data/test/x_test.txt')
    y_test = load_data_sample('data/test/y_test.txt')

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = [predict_language(text, model) for text in X_test]
    print("Accuracy:", accuracy_score(y_test, y_pred))