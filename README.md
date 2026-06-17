# Fake Job Detection System Using NLP and Machine Learning

This project is a machine learning-based web application that detects whether a job posting is real or fake using Natural Language Processing.

## Project Objective

The main objective of this project is to identify fake job postings from job descriptions. Fake job posts can mislead job seekers, so this system helps classify job descriptions as real or fake.

## Dataset

The project uses the Fake Job Postings dataset.

Important columns used:

- `description` - job description text
- `fraudulent` - target column

Target labels:

- `0` = Real Job
- `1` = Fake Job

## Technologies Used

- Python
- Pandas
- NumPy
- Regex
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Multinomial Naive Bayes
- Flask
- HTML/CSS
- Joblib

## Project Workflow

1. Loaded the dataset.
2. Selected job description and target columns.
3. Removed missing values.
4. Cleaned text using Regex.
5. Performed feature engineering.
6. Converted text into numerical vectors using TF-IDF.
7. Split the data into training and testing sets.
8. Trained machine learning models.
9. Compared model performance.
10. Selected the best model.
11. Saved the trained model and vectorizer using Joblib.
12. Built a Flask web application for prediction.

## Text Preprocessing

The job description text was cleaned by:

- Converting text to lowercase
- Removing URLs
- Removing HTML tags
- Removing special characters
- Removing extra spaces

## Models Used

- Multinomial Naive Bayes
- Logistic Regression

Logistic Regression was selected as the final model because it performed better for classification and handled the imbalanced dataset better.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The final model achieved good accuracy and improved fake job detection recall.

## 🔗 Live Demo

https://fake-job-detection-tau.vercel.app/


