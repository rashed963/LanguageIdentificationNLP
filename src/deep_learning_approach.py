import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from datasets import load_metric
import torch

def load_data(filepath, tokenizer, max_length=512):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    encodings = tokenizer(lines, truncation=True, padding=True, max_length=max_length)
    return torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask'])

def model_init(num_labels):
    return DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=num_labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

if __name__ == "__main__":
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-multilingual-cased")
    
    # Load data
    X_train_ids, X_train_mask = load_data('data/train/x_train.txt', tokenizer)
    y_train = np.loadtxt('data/train/y_train.txt', dtype=int)
    X_test_ids, X_test_mask = load_data('data/test/x_test.txt', tokenizer)
    y_test = np.loadtxt('data/test/y_test.txt', dtype=int)
    
    # Convert labels to torch tensors
    y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
    
    # Model initialization
    model = model_init(num_labels=len(set(y_train.tolist() + y_test.tolist())))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        evaluate_during_training=True,
        logging_dir='./logs',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=({'input_ids': X_train_ids, 'attention_mask': X_train_mask, 'labels': y_train}),
        eval_dataset=({'input_ids': X_test_ids, 'attention_mask': X_test_mask, 'labels': y_test}),
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluation
    results = trainer.evaluate()
    print("Test Accuracy:", results["eval_accuracy"])