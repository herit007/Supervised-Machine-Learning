# 🏠 House Price Prediction Using Machine Learning

A Machine Learning project that predicts house prices using multiple regression models and compares their performance using various evaluation metrics.

---

## 📖 Overview

This project focuses on predicting house prices based on property characteristics such as area, number of bedrooms, location score, property age, distance from the city, nearby facilities, and crime rate index.

The main objective is to compare different machine learning models and identify the most accurate model for house price prediction.

---

## 📊 Dataset Information

**Dataset Size:** 3800 Rows × 12 Columns

### Features

* Property ID
* Sale Date
* Area (sqft)
* Bedrooms
* Bathrooms
* Location Score
* Property Age
* Distance from City (km)
* Near School
* Near Metro
* Crime Rate Index

### 🎯 Target Variable

* House Price (INR)

---

## 🔄 Project Workflow

### 1️⃣ Data Understanding

* Dataset Inspection
* Missing Value Analysis
* Duplicate Value Check
* Statistical Summary

### 2️⃣ Exploratory Data Analysis (EDA)

* House Price Distribution
* Correlation Heatmap
* Feature Relationship Analysis

### 3️⃣ Data Preprocessing

* Date Conversion
* Feature Extraction
* Removal of Unnecessary Columns
* Train-Test Split (80:20)
* Feature Scaling

### 4️⃣ Model Building

* Ridge Regression
* Lasso Regression
* Decision Tree Regression
* Random Forest Regression
* Support Vector Regression (SVR)

### 5️⃣ Cross Validation

* K-Fold Cross Validation
* Stratified K-Fold Cross Validation
* Leave-One-Out Cross Validation (LOOCV)
* Time Series Split

### 6️⃣ Model Evaluation

The following metrics were used:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

## 🤖 Models Used

| Model                    | Purpose                            |
| ------------------------ | ---------------------------------- |
| Ridge Regression         | Regularized Linear Model           |
| Lasso Regression         | Feature Selection & Regularization |
| Decision Tree Regression | Tree-Based Learning                |
| Random Forest Regression | Ensemble Learning                  |
| SVR (Linear Kernel)      | Support Vector Regression          |
| SVR (RBF Kernel)         | Non-Linear Regression              |

---

## 📈 Results

| Model                    | R² Score |
| ------------------------ | -------- |
| Ridge Regression         | 0.92     |
| Lasso Regression         | 0.92     |
| Decision Tree Regression | 0.88     |
| Random Forest Regression | 0.93     |
| SVR Linear               | 0.01     |
| SVR RBF                  | -0.00    |

---

## 🏆 Best Performing Model

### Random Forest Regression

✅ Highest R² Score (0.93)

✅ Lowest Prediction Error

✅ Best Overall Performance

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## 💡 Conclusion

Different machine learning models were trained and evaluated for house price prediction. Based on the evaluation metrics, **Random Forest Regression** achieved the highest R² Score and the lowest prediction error, making it the most suitable model for this dataset.

---

## 👨‍💻 Author

**Herit**
