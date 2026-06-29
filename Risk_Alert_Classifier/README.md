# 🚨 Risk Alert Classifier

A Machine Learning classification project that predicts whether a customer is **High Risk** or **Low Risk** using demographic, financial, and behavioral information. This project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, exploratory data analysis (EDA), handling class imbalance, model building, hyperparameter tuning, and performance evaluation.

---

## 📌 Project Overview

Financial institutions need to identify high-risk customers to reduce loan defaults and financial losses. Since high-risk customers are often a minority class, this project also explores various resampling techniques to improve model performance on imbalanced data.

---

## 🎯 Objectives

- Understand and explore the dataset.
- Perform data cleaning and preprocessing.
- Handle missing values using **KNN Imputer**.
- Perform Exploratory Data Analysis (EDA).
- Handle imbalanced datasets using different sampling techniques.
- Build multiple classification models.
- Compare model performance using different evaluation metrics.
- Identify the best-performing model for risk prediction.

---

## 📂 Dataset Information

The dataset contains customer demographic, financial, and behavioral attributes.

### Features

- Customer ID
- Age
- Gender
- Region
- Employment Type
- Annual Income
- Credit Score
- Credit Utilization Ratio
- Missed Payments (Last 12 Months)
- Average Late Payment Days
- Monthly Transaction Count
- Monthly Spending
- Cash Advance Count
- Complaints (Last 6 Months)
- Failed Login Attempts
- Account Tenure
- Last Transaction Date
- Debt Balance

### Target Variable

- **Risk Status**
  - Low Risk
  - High Risk

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

---

# 📊 Project Workflow

```

Dataset
│
▼
Data Cleaning
│
▼
Missing Value Handling
(KNN Imputer)
│
▼
Exploratory Data Analysis
│
▼
Encoding
│
▼
Feature Scaling
│
▼
Train-Test Split
│
▼
Model Building
│
▼
Evaluation

```

---

# 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

# ⚖ Handling Imbalanced Data

The following sampling techniques are implemented:

- Random Under Sampling
- Random Over Sampling
- SMOTE
- ADASYN

---

# 📈 Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Risk_Alert_Classifier.git
````

Move into the project folder:

```bash
cd Risk_Alert_Classifier
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook
```

---

# 📚 Python Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

---

# 📌 Key Learning Outcomes

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Missing Value Imputation
* Feature Scaling
* Handling Imbalanced Data
* Classification Algorithms
* Hyperparameter Tuning
* Model Comparison
* Machine Learning Model Evaluation

---

## 👨‍💻 Author

**Herit Tanna**