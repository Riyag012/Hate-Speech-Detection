# Hate Speech Detection

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Performance](#model-performance)

## Overview

This project aims to detect **hate speech** in text using machine learning techniques. By leveraging natural language processing (NLP) and a **Decision Tree Classifier**,
it identifies and categorizes text as either **hate speech**, **offensive language**, or **neither**. The project uses a **tweets dataset from Kaggle** as the primary
data source, enabling the analysis of real-world social media data.

The goal of this project is to contribute to creating a safer and more inclusive online space by detecting harmful language automatically. It highlights the potential
of machine learning in addressing issues of toxicity and hate speech in digital communication.

## Features

- **Data Collection**: Dataset obtained from Kaggle.
- **Text Preprocessing**: Removal of stop words, punctuation, and special characters.
- **Vectorization**: Conversion of text into numerical representations using CountVectorizer.
- **Model Training**: Classification using **Decision Tree Classifier**.
- **Evaluation**: Metrics such as accuracy, precision, recall, and F1-score to evaluate model performance.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: 
  - `scikit-learn`
  - `pandas`
  - `nltk`
  - `matplotlib`
  - `seaborn`
- **Model**: DecisionTreeClassifier
- **Preprocessing Techniques**: Tokenization, Stop word removal, Lemmatization

## Dataset

- The dataset contains text samples categorized as:
  - **Hate Speech**
  - **Offensive Language**
  - **Neither**
- It is sourced from Kaggle: [Hate Speech Dataset](https://www.kaggle.com/datasets).

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd hate-speech-detection
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset with **text** and **label** columns.
2. Train the model by running:
   ```bash
   python train_model.py
   ```
3. Use the model to make predictions:
   ```bash
   python predict.py --input "Your input text here"
   ```

## Model Performance

The model achieved the following performance metrics:
- **Accuracy**: 87.55%
- **Precision**: 87.29%
- **Recall**: 87.55%
- **F1 Score**: 87.42%

These metrics were computed using the testing dataset and demonstrate the model's effectiveness in detecting hate speech and offensive language.

