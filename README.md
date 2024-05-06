# Language Identification NLP

## Project Overview

This repository contains implementations of three distinct approaches for language identification, leveraging both classical machine learning and advanced deep learning techniques. Each approach is designed to understand and classify text based on linguistic patterns across multiple languages.

## Dataset
The models were trained and tested using the "Wili-2018" dataset for language identification, which includes samples from 235 languages:

Training Data: 25,000 samples
Testing Data: 10,000 samples
This subset was chosen due to computational resource limitations, ensuring the models are both effective and efficient.


## Approaches

Approach 1: Character-Level N-Gram and Logistic Regression
Utilizes character-level n-grams to capture language-specific character patterns.
Employs logistic regression to classify text into one of the 235 languages.

Model and tokenizer files are saved under ./approach_1_model/.


Approach 2: TF-IDF and Word2Vec with Grid Search Optimization
Features two vectorization techniques: TF-IDF and Word2Vec, to represent text data.
Applies Grid Search to fine-tune logistic regression parameters for optimal performance.

Model and tokenizer files are saved under ./approach_2_model/.


Approach 3: Fine-Tuning BERT for Language Identification
Fine-tunes a pre-trained BERT model to leverage deep contextual representations.
Specifically uses the 'bert-base-multilingual-cased' model to support multiple languages effectively.

Model and tokenizer files are saved under ./approach_3_model/.


## Results
The final evaluation of models on the test dataset is presented as follows:

Approach 1
#model_v0.1:  ngram_range=(1, 1): iter = 50: Accuracy: 0.8085
#model_v0.2:  ngram_range=(1, 1): iter = 500: Accuracy: 0.8513

Approach 2
#mode_lr_tfidf_bigram_best, ngram_range=(2, 2), iter = 500, tfidf, max_iter=500, Accuracy: 0.9274

Approach 3
for 2 epochs: the eval_loss:0.25

## General Analysis
Effectiveness of N-Grams: The results highlight the impact of using different n-gram ranges and the importance of iteration numbers in model convergence and performance. Single grams provide a baseline, while bigrams significantly enhance model capability by incorporating more contextual cues.

TF-IDF Strengths: The success of the TF-IDF model in Approach 2 suggests that statistical feature weighting still plays a crucial role in text classification tasks, particularly when paired with effective optimization strategies like grid search.

Deep Learning Potential: Approach 3's BERT model, despite lacking direct accuracy comparison, likely represents a more advanced text understanding capability due to its deep learning nature and the ability to model complex language patterns. However, quantifiable metrics are needed for a fair comparison.


Resource Considerations: It's evident that computational resources and training iterations play a significant role in the effectiveness of each model. As such, balancing resource expenditure with model performance is key, especially in scenarios with limited computational capacity.



## Models Download

The trained trained in this project are available for download external drive. Follow the link below to download the latest version of the pre-trained models:

(https://drive.google.com/drive/folders/1CiVLdwLYDQjuBo1Rk5bOuF6LqdYHjjyU?usp=sharing)

### Instructions
- Download the `models.zip` file from the latest release.
- Extract the zip file and replace the `models` directory in this project with the extracted folder.
