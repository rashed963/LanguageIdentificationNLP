import random
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score

def load_data_sample(filepath, sample_size=1000, random_state=42):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.seed(random_state)
    sampled_indices = random.sample(range(len(lines)), sample_size)
    sampled_lines = [lines[i].strip() for i in sampled_indices]
    return sampled_lines

def load_pretrained_model(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict_language(text, model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**encoded_input)
    logits = outputs.logits
    predicted_label = torch.argmax(logits)
    return predicted_label.item()

if __name__ == "__main__":
    # Load data
    X_train = load_data_sample('data/train/x_train.txt', sample_size=1000)
    y_train = load_data_sample('data/train/y_train.txt', sample_size=1000)
    X_test = load_data_sample('data/test/x_test.txt')
    y_test = load_data_sample('data/test/y_test.txt')

    unique_labels = list(set(y_train))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    y_train_tensor = torch.tensor([label_to_id[label] for label in y_train])
    # Load pre-trained model and tokenizer
    model_name = 'distilbert-base-uncased'
    num_labels = len(set(y_train))
    model, tokenizer = load_pretrained_model(model_name, num_labels)

    # Fine-tune the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Check data and model architecture
    print("Data shapes:")
    print("X_train:", pd.DataFrame(X_train).shape)
    print("y_train:", pd.DataFrame(y_train).shape)
    print("X_test:", pd.DataFrame(X_test).shape)
    print("y_test:", pd.DataFrame(y_test).shape)

    print("\nData samples:")
    print("X_train[0]:", X_train[0])
    print("y_train[0]:", y_train[0])
    print("X_test[0]:", X_test[0])
    print("y_test[0]:", y_test[0])

    print("\nModel architecture:")
    print(model)

    print("\nClassification head:")
    print(model.classifier)

    print("\nOptimizer and loss function:")
    print(optimizer)
    print(loss_fn)




for epoch in range(1):
    model.train()
    optimizer.zero_grad()

    # Prepare inputs
    encoded_input = tokenizer(X_train, 
                            return_tensors='pt', 
                            max_length=256, 
                            truncation=True, 
                            padding='max_length')

    # Forward pass
    outputs = model(**encoded_input)

    # Extract logits tensor from SequenceClassifierOutput
    logits = outputs.logits

    # Prepare labels
    y_train_tensor = torch.tensor([label_to_id[label] for label in y_train])

    # Calculate loss
    loss = loss_fn(logits, y_train_tensor)

    # Backward pass
    loss.backward()

    # Update model parameters
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    y_pred = [predict_language(text, model, tokenizer) for text in X_test]
    print("Accuracy:", accuracy_score(y_test, y_pred))
