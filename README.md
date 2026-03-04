# 🚀 Early Warning Prediction System (Customer Churn Prediction)

An end-to-end Machine Learning system that predicts customer churn using classification models and provides explainable insights to support proactive customer retention strategies.

---

## 📌 Project Overview

The Early Warning Prediction System is designed to identify customers who are likely to churn. By analyzing historical customer behavior data, the system predicts churn probability and provides interpretable explanations using SHAP values.

This project follows a complete ML lifecycle:

- Data Collection
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training & Evaluation
- Model Explainability
- Deployment (Flask & Streamlit)
- Threshold Optimization

---

## 🎯 Business Problem

Customer churn significantly impacts revenue and growth. 

Instead of reacting after customers leave, this system enables:

- Early risk detection
- Targeted retention strategies
- Reduction in churn rate
- Improved customer lifetime value (CLV)

---

## 🧠 Technical Architecture

Raw Data → Data Cleaning → Feature Engineering → Model Training
→ Model Evaluation → Threshold Selection
→ Model Serialization → Web App Deployment
→ Real-Time Prediction + SHAP Explanation


---

## 🛠 Tech Stack

### 💻 Programming
- Python 3.x

### 📊 Data Handling
- Pandas
- NumPy

### 🤖 Machine Learning
- Scikit-learn
- Logistic Regression / Tree-based model
- SHAP (Model Explainability)

### 🌐 Deployment
- Flask (Web application)
- Streamlit (Alternative UI)
- Gunicorn (Production WSGI server)
- Heroku-ready setup (Procfile + runtime.txt)

### 📦 Model Persistence
- Pickle (.pkl files)

---

## 📂 Project Structure

Early_Warning_Prediction/
│
├── app.py # Flask application
├── stream_app.py # Streamlit version
├── requirements.txt # Dependencies
├── Procfile # Deployment config
├── runtime.txt # Python runtime
│
├── data/
│ ├── raw_churn.csv
│ ├── cleaned.csv
│ ├── cleaned_churn.csv
│ └── sample.csv
│
├── model/
│ ├── churn_model.pkl
│ ├── explainer.pkl
│ └── threshold.pkl
│
├── src/
│ ├── train.py
│ ├── preprocess.py
│ ├── predict.py
│ ├── main.py
│ └── config.py
│
├── templates/
│ ├── index.html
│ └── dashboard.html
│
└── static/
└── css/style.css


---

## 🔎 Key Features

### ✅ 1. Data Preprocessing
- Handling missing values
- Feature encoding
- Feature scaling
- Data cleaning pipeline

### ✅ 2. Model Training
- Supervised classification
- Train-test split
- Hyperparameter tuning
- Model evaluation (Accuracy, Precision, Recall, F1-score)

### ✅ 3. Threshold Optimization
Rather than using default 0.5 threshold:
- Custom threshold saved in `threshold.pkl`
- Business-driven decision boundary

### ✅ 4. Model Explainability (SHAP)
- Explains prediction at individual customer level
- Identifies key features contributing to churn
- Increases transparency and trust

### ✅ 5. Deployment
- Flask-based production-ready web app
- Streamlit quick interactive version
- Heroku deployment compatible

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

Business priority was given to:
> Higher Recall → Minimize false negatives (reduce missed churn cases)

---

## 🧮 Model Explainability

We used SHAP to:

- Measure global feature importance
- Explain individual predictions
- Visualize impact of each feature

This ensures the model is not a black box.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone <repo-url>
cd Early_Warning_Prediction

Install Dependancies
pip install -r requirements.txt

Run Flask App
python app.py

Run Streamlit App
streamlit run stream_app.py

## 🧠 Technical Architecture

