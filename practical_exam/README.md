# 📉 Customer Churn Prediction
### Multi-Algorithm Classification Showdown

A supervised machine learning project that predicts whether a telecom customer is likely to churn, built using **KNN, Naive Bayes, SVM, and Decision Tree** classifiers with full preprocessing, class imbalance handling, and a deployable sklearn Pipeline.

---

## 🎥 Demo Video

Watch the full project walkthrough here:
**[▶ Project Explanation Video](https://drive.google.com/file/d/1shCxJVfu4y0QSdLSw_TW2y4v4fQabUkV/view?usp=sharing)**

---

## 🔗 Quick Links

| Resource | Location |
|---|---|
| 📓 Jupyter Notebook | [`Notebook/CustomerChurn_SupervisedLearning.ipynb`](https://github.com/herit007/Supervised-Machine-Learning/blob/main/practical_exam/Notebook/CustomerChurn_SupervisedLearning.ipynb) |
| 🤖 Saved Model | [`Notebook/churn_model.pkl`](https://github.com/herit007/Supervised-Machine-Learning/blob/main/practical_exam/Notebook/churn_model.pkl) |
| 📊 Dataset | [`Dataset/Telco-Customer-Churn.csv`](https://github.com/herit007/Supervised-Machine-Learning/blob/main/practical_exam/Dataset/Telco-Customer-Churn.csv) |
| 📋 Summary Report | [`summary_report.md`](https://github.com/herit007/Supervised-Machine-Learning/blob/main/practical_exam/summary_report.md) |
| 📦 Requirements | [`requirements.txt`](https://github.com/herit007/Supervised-Machine-Learning/blob/main/practical_exam/requirements.txt) |

---

## 📌 Project Overview

**Customer Churn Prediction** is a data science project built for a telecom company scenario. The goal is to identify customers who are likely to leave the service so the retention team can intervene before it happens.

The dataset contains customer demographic info, subscribed services, contract type, billing details, and whether each customer churned. The project compares four classification models to find the best one for flagging high-risk customers.

| Target | Meaning |
|---|---|
| `0 → No Churn` | Customer is retained |
| `1 → Churn` | Customer has left the service |

---

## 🎯 Objective

To design a complete churn prediction pipeline that:

- Accurately identifies high-risk customers before they churn
- Handles the class imbalance in the dataset using SMOTE
- Evaluates models using metrics that actually matter for this business problem
- Deploys the best model as a reusable sklearn Pipeline saved as a `.pkl` file

---

## 💡 Business Context

| Concept | Explanation |
|---|---|
| **CAC (Customer Acquisition Cost)** | What the company spends to bring in a new customer — 5–10x higher than retention cost |
| **CLV (Customer Lifetime Value)** | Total revenue a customer generates over their lifetime with the company |
| **False Negative (FN)** | Actual churner predicted as safe → customer leaves undetected → most costly mistake |
| **False Positive (FP)** | Safe customer flagged as churner → one unnecessary retention call → small fixed cost |

Since a False Negative is far more costly than a False Positive, **Recall** is the primary evaluation metric.

---

## 🗂️ Dataset

**IBM Telco Customer Churn Dataset**

| Property | Detail |
|---|---|
| Source | [Kaggle — IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows | 7,043 customers |
| Features | 21 columns |
| Target Column | `Churn` — Yes / No |
| Churn Rate | ~26.5% — imbalanced dataset |

**Feature Categories:**

| Category | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, OnlineSecurity, StreamingTV, etc. |
| Contract & Billing | Contract, PaperlessBilling, PaymentMethod |
| Charges | MonthlyCharges, TotalCharges, tenure |

---

## 🧠 What This Project Does

- Writes theory answers as markdown cells directly inside the notebook (churn definition, confusion matrix, SMOTE, all 4 algorithms, Precision vs Recall)
- Performs EDA on a dataframe copy (`df_eda`) to keep raw data untouched
- Fixes the `TotalCharges` column which loads as object due to whitespace in 11 rows
- Engineers 3 new features — `tenure_group`, `num_services`, `AutoPay`
- Applies binary encoding, one-hot encoding and label encoding correctly
- Performs 80/20 stratified train-test split and applies SMOTE only on the training set
- Implements and tunes all 4 classifiers with confusion matrices and ROC curves
- Compares SMOTE vs `class_weight='balanced'` for handling imbalance
- Analyzes False Negatives to understand which churners the model misses
- Saves a full sklearn Pipeline as `churn_model.pkl` — auto-created when notebook runs

---

## ⚙️ Project Workflow

| Step | What was done |
|---|---|
| **Step 1 — Theory** | 5 theory answers as markdown cells — churn + CAC/CLV, confusion matrix, SMOTE, 4 algorithms, Precision vs Recall |
| **Step 2 — EDA** | TotalCharges fix, class balance check, histograms, countplots, churn by contract, churn by tenure bucket, boxplot, heatmap |
| **Step 3 — Preprocessing** | df.copy(), drop customerID, fill nulls, feature engineering, encoding, split, scaling, SMOTE |
| **Step 4 — KNN** | Baseline k=5, k tuning (1,3,5,7,9,11,15), confusion matrix, ROC curve |
| **Step 4 — Naive Bayes** | GaussianNB, class priors check, confusion matrix, ROC curve, independence note |
| **Step 5 — SVM** | Baseline + C tuning (3-fold CV), confusion matrix, ROC curve, black-box note |
| **Step 5 — Decision Tree** | Baseline + tree visualization + depth tuning + top 15 feature importances + SMOTE vs class_weight |
| **Step 6 — Comparison** | Table sorted by Recall, all 4 ROC curves, Precision vs Recall bar chart, recommendation |
| **Step 7 — Error Analysis** | False Negative profiling, churner profile comparison, top 5 business churn signals |
| **Step 8 — Pipeline** | sklearn Pipeline saved as pkl, loaded and tested on 5 sample customers |

---

## 📊 Model Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **KNN** | 0.7154 | 0.4785 | **0.8021** | 0.5994 | 0.8121 |
| **Naive Bayes** | 0.7232 | 0.4869 | 0.7941 | 0.6037 | 0.8164 |
| **Decision Tree** | 0.7530 | 0.5240 | 0.7594 | **0.6201** | 0.8052 |
| **SVM** | 0.7239 | 0.4863 | 0.7112 | 0.5776 | 0.7884 |

> All metrics are for the **Churn class (class = 1)**, table sorted by Recall (descending).

---

## 🔍 Key Churn Signals Found

| Signal | Finding | Action |
|---|---|---|
| **Contract Type** | Month-to-Month customers churn 3–4x more than yearly | Offer discounted annual plan upgrade |
| **Tenure < 12 months** | Churn rate above 40% in first year | Proactive onboarding + check-in calls |
| **High MonthlyCharges** | Higher bills = higher churn risk | Loyalty pricing for expensive plan holders |
| **No AutoPay** | Manual payers reconsider their bill each month | Encourage autopay enrollment |
| **Low num_services** | Fewer add-ons = lower switching cost | Offer free add-on trial to increase stickiness |

---

## 🚀 Recommended Model for Deployment

**Highest Recall → KNN** (0.80) — catches the most actual churners

**Best for Production → Decision Tree** because:
- Best overall F1-Score (0.6201)
- Fully interpretable — team can read the rules and understand why a customer was flagged
- Fast prediction — no distance computation overhead at scale
- Feature importances tell the team exactly which attribute to address in each retention call

**Using the saved pipeline:**
```python
import joblib
pipeline = joblib.load('churn_model.pkl')

probabilities = pipeline.predict_proba(new_customer_data)[:, 1]
predictions   = pipeline.predict(new_customer_data)
```

---

## 🛠️ Tech Stack & Libraries

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.x |
| Data Handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (KNN, Naive Bayes, SVM, Decision Tree) |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Model Persistence | joblib |
| Environment | Jupyter Notebook |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/herit007/Supervised-Machine-Learning.git
cd Supervised-Machine-Learning/practical_exam

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook Notebook/CustomerChurn_SupervisedLearning.ipynb
```

> Running all cells automatically creates `churn_model.pkl` inside the `Notebook/` folder.

---

## 📁 Folder Structure

```
practical_exam/
│
├── Dataset/
│   └── Telco-Customer-Churn.csv
│
├── Notebook/
│   ├── CustomerChurn_SupervisedLearning.ipynb
│   └── churn_model.pkl
│
├── README.md
├── requirements.txt
└── summary_report.md
```

---

## 👤 Author

**Herit Tanna**
Computer Engineering Diploma — A.V.P.T.I. Rajkot (GTU)
AI/ML & Data Science — Red & White Multimedia Education

---

⭐ If you found this project useful, consider giving it a star!
