import numpy as np
import random
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_and_sample_data(filepath, tokenizer, sample_size=1000, max_length=128, random_state=42):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.seed(random_state)
    sampled_indices = random.sample(range(len(lines)), min(sample_size, len(lines)))
    sampled_texts = [lines[i].strip() for i in sampled_indices]

    encodings = tokenizer(sampled_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    return encodings, sampled_indices

def load_labels(filepath, sampled_indices):
    with open(filepath, 'r', encoding='utf-8') as file:
        labels = [line.strip() for line in file.readlines()]

    sampled_labels = [labels[i] for i in sampled_indices]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(sampled_labels)
    return encoded_labels, label_encoder

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def model_init(num_labels):
    return DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=num_labels)

if __name__ == "__main__":
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-multilingual-cased")
    
    # Load and sample data
    X_train_encodings, train_indices = load_and_sample_data('data/train/x_train.txt', tokenizer)
    y_train, label_encoder = load_labels('data/train/y_train.txt', train_indices)
    
    X_test_encodings, test_indices = load_and_sample_data('data/test/x_test.txt', tokenizer)
    y_test, _ = load_labels('data/test/y_test.txt', test_indices)

    # Convert to dataset
    train_dataset = TextDataset(X_train_encodings, y_train)
    eval_dataset = TextDataset(X_test_encodings, y_test)

    # Model initialization
    model = model_init(num_labels=len(label_encoder.classes_))

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluation
    results = trainer.evaluate()
    print("Test Accuracy:", results["eval_accuracy"])