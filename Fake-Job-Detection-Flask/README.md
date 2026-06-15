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

## Files in This Project

```text
Fake-Job-Detection-
│
├── app.py
├── templates/
│   └── index.html
├── fake_job_postings.csv
├── Fakejobdetection.ipynb
├── model_fake_job.pkl
├── vectorizer_fake_job.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/NityaGutti2468/Fake-Job-Detection-.git
cd Fake-Job-Detection-
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 5: Run Flask App

```bash
python app.py
```

Open this URL in browser:

```text
http://127.0.0.1:5000/
```

## Sample Prediction Flow

1. User enters a job description.
2. Text is cleaned.
3. TF-IDF vectorizer converts text into numerical features.
4. ML model predicts whether the job is real or fake.
5. Result is shown on the web page.

## Interview Explanation

I developed a Fake Job Detection System using NLP and Machine Learning. The system uses job descriptions as input and predicts whether the job posting is real or fake. I cleaned the text using Regex, converted the text into numerical features using TF-IDF, and trained models like Multinomial Naive Bayes and Logistic Regression. Logistic Regression was selected as the final model because it gave better performance. I saved the trained model and vectorizer using Joblib and built a Flask web application for real-time prediction.

## Future Improvements

- Add more job-related columns like title, company profile, and requirements
- Improve model performance using advanced NLP techniques
- Deploy the Flask app online
- Add login and prediction history
