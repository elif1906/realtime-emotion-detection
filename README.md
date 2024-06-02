# Real-Time Emotion Detection (Face - Voice)

By student : Elif Nur Aslıhan Celepoğlu

Id:1904010023

<img src="image.png" alt="alt text" width="900" height="600">



This project aims to develop a system for analyzing real-time facial expressions and voices. The system detects emotions as a result of analyzing facial expressions and sounds. 
This project was developed as a web project and allows users to detect their emotions by analyzing their facial expressions and voices.

## Introduction

- used dataset fer-2023 and RoBERTa
- fer-2023 used Vision Transformer (ViT)
- RoBERTa used Masked language modeling (MLM) 

**seven emotions** - angry, disguste, fear, happy, neutral, sad and surprised.

## Result output 
<img src="result.png" alt="alt text" width="900" height="600">

## Installation Instructions

### First, clone the repository and enter the folder

For Windows;
```bash
git clone https://github.com/elif1906/realtime-emotion-detection.git
cd realtime-emotion-detection
```
 For Mac;
 ```bash
git clone git@github.com:elif1906/realtime-emotion-detection.git
cd realtime-emotion-detection
```
### Second, requirements.txt


```bash
pip install -r  requirements.txt
```

### Third, run app.py file
```bash
python app.py
```

## About the Models
-Face Model: The images were processed as follow:(fer2023)
 -Data Processing:
The input images are preprocessed before being fed into the model. The preprocessing steps include:

Resizing: Images are resized to the specified input size.
Normalization: Pixel values are normalized to a specific range.
Data Augmentation: Random transformations such as rotations, flips, and zooms are applied to augment the training dataset.

 -Modeling:

ViT (Vision Transformer) Facial Expression Recognition Model

This model is based on the Transformer architecture, which has recently become popular in the field of image processing. Transformers have been successfully used in language models (for text data), but they have also been adapted to the field of image processing. Vision Transformer (ViT) is an example of this transformation.

   -Model Features:
ViT utilizes Transformer blocks instead of convolutional neural networks (CNNs) for image processing tasks.
This model has been trained for facial expression recognition tasks.
The training data consists of facial photographs representing different emotional expressions.
The model can be used to recognize and classify emotional expressions.

<img src="vit.png" alt="alt text" width="900" height="600">


 -Training Data:
Evaluation Metrics
Validation set accuracy: 0.7113
Test set accuracy: 0.7116


-Voice Model: The waves were processed as follow:(RoBERTa)

  -Data Processing:

While training, the goal is to first create start and end logits for each context for a particular question. The start index is already given and using the length of the answer, the end index is determined.

We then make a mask(list) which is the size of the context. Those positions which contain the answer are given the value of 1 whereas others are given the value of 0.

The next step is to create offsets for each token in the context and appending those tokens whose offset span contain ones to a list. This list now contains the tuples holding the positions of those tokens present in the answer. Thus, the start token becomes the first tuple in the list whereas the end token becomes the last tuple in the list.

  -Modeling:
 Masked Language Modeling (MLM) is a technique used to train language models. Essentially, by hiding (masking) some words or tokens within the text, it allows the model to predict these hidden words. This method allows the model to better understand the context of the language and thus make more accurate predictions. 
 My aim here is to use a method that performs sentiment analysis from text, and I also added sound. In other words, it converts the audio into text and then performs sentiment analysis.
<img src="mlm.png" alt="alt text" width="900" height="600">


  -Training:
Learning Rate = 3e-5 (As specified in the paper) Loss - categorical_crossentropy Optimizer - Adam Batch Size - 4(Due to computational limitations. Not: For optimal results, as specified in the bert paper, batch size must be equal to 32 or 64) Epochs = 3/4 (As specified in the paper)


