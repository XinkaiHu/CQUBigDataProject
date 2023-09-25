# Distributed Neural-Network-Based News Headline Categorization

overall introduction: ...

# Dataset

We used NLPCC 2017 (Task 2) dataset to train and evaluate our model.

The NLPCC2017 dataset is divided into a training set of 156'000 Chinese news headlines and a test set of 36,000 Chinese news headlines. Both of the training and test sets have 18 classes: ...

Figures about the dataset...

# Data Processing

Overall steps: segmentation, stopwords, vocabulary, word embedding...

## Segmentation

### By words

### By characters

We use `jieba` library in Python to segment sentences.

Introduction to `jieba`...

Introduction to different `jieba` segmentation modes: ...

## Stopwords

### Drop stopwords

Why drop stopwords: a general procedure in text classification tasks...

### Reserve stopwords

Since news headlines are **short text**s, stopwords are less frequently appearing than long texts...

## Vocabulary and Word Embedding

What is word embedding...

# Model

We firstly built a simple neural network (Model 1) composed with an embedding layer and a single full connected layer...

# Results

On Model 1, after about 40 epochs, the accuracy on test set stopped decreasing at 0.733, slightly less than the baseline results...

The loss curve...

Accuracy, Micro-P, Micro-R, Macro-P, Macro-R, F_1 score...

# Acknowledgement
...