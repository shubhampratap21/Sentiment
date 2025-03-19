# **Tweet Sentiment Analysis**

A machine learning project that analyzes and predicts the sentiment of tweets. This implementation uses TF-IDF vectorization and a Linear Support Vector Classifier (SVC) to classify tweets into various emotion categories such as happiness, sadness, anger, etc.

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Performance](#model-performance)
7. [Future Enhancements](#future-enhancements)
8. [Contributing](#contributing)

---

## **Introduction**

This project implements a sentiment analysis model to classify emotions in tweets. The model leverages:
- TF-IDF vectorization for text preprocessing.
- Linear Support Vector Classifier (SVC) for classification.

It provides functionality for both batch evaluation on a dataset and single-sentence predictions via user input.

---

## **Features**

- **Text Preprocessing**: Removes stopwords, lowers case, and tokenizes the tweets.
- **Model Training**: Uses a LinearSVC classifier trained on TF-IDF features.
- **Batch Evaluation**: Evaluates accuracy, precision, recall, and f1-score on a test dataset.
- **Single Input Prediction**: Accepts user-provided tweets and predicts the sentiment in real-time.
- **Confusion Matrix**: Visualizes the model's performance on the test dataset.

---

## **Dataset**

The dataset used for this project is `tweet_emotions.csv`. It contains:
- `content`: The text of the tweet.
- `sentiment`: The label corresponding to the sentiment (e.g., happiness, sadness, etc.).

Sample entry:
```
tweet_id,sentiment,content
1956967341,empty,@tiffanylue I know I was listening to bad habit earlier and I started freaking at his part =[
```

---

## **Installation**

### Prerequisites

- Python 3.7+
- Required libraries: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/tweet-sentiment-analysis.git
   cd tweet-sentiment-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the `tweet_emotions.csv` dataset in the project directory.

---

## **Usage**

### Run the Sentiment Analysis Model

1. Train and test the model:
   ```bash
   python sentiment_analysis.py
   ```

2. After training, you can input a tweet to predict its sentiment:
   ```
   Enter a tweet (or type 'exit' to quit): I am so happy today!
   Predicted Sentiment: happiness
   ```

---

## **Model Performance**

- **Accuracy**: ~90.4%
- **Confusion Matrix**: The confusion matrix is displayed as a heatmap after training.

Example of classification report:
```
               precision    recall  f1-score   support
   happiness       0.29      0.31      0.30      3160
   sadness         0.28      0.27      0.28      3106
   ...
```

---

## **Future Enhancements**

- Improve accuracy using advanced models like BERT or RoBERTa.
- Add more preprocessing steps (e.g., stemming, lemmatization).
- Allow model saving and loading for faster predictions without retraining.
- Support additional datasets for broader generalization.

---

## **Contributing**

Contributions are welcome! Feel free to:
- Fork the repository.
- Create a branch for your feature.
- Submit a pull request.

---

