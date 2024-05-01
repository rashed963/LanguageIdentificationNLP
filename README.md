# Language Identification NLP
## Naive model


## Pretrained Model Approach

In pretrained_model_approach.py, I implemented a fine-tuning approach using a pre-trained DistilBERT model for text classification. The script loads a sample dataset, fine-tunes the pre-trained model on the training data, and evaluates its performance on the testing data.

The implementation involves the following steps:

Loading a sample dataset consisting of text inputs and corresponding language labels
Fine-tuning a pre-trained DistilBERT model using a stochastic gradient descent (SGD) optimizer and cross-entropy loss function.
Evaluating the model's performance on the testing data using accuracy as the evaluation metric

Note: Due to computational limitations, I was unable to complete the execution of the script and obtain the final results. The script is provided as a starting point, and further work is needed to complete the fine-tuning process and evaluate the model's performance.

## References
- "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" by Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf (2020)
- https://huggingface.co/docs/transformers/index
