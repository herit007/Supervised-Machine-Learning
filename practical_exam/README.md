# Customer Churn Prediction
### Multi-Algorithm Classification Showdown

> Predicting which telecom customers are about to leave — so the retention team can act before it's too late.

---

## Business Problem

Every month a telecom company loses paying subscribers. The cost of acquiring a new customer **(CAC)** is 5–10x higher than the cost of retaining an existing one. When a customer churns, the company loses their entire remaining **Customer Lifetime Value (CLV)** and has to spend again to replace them.

This project builds and compares four classification models — **KNN, Naive Bayes, SVM, and Decision Tree** — to flag high-risk customers before they churn, giving the retention team a daily prioritized list to act on.

---

## Dataset

**IBM Telco Customer Churn Dataset**

| Property | Detail |
|---|---|
| Source | [Kaggle — IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 columns |
| Target | `Churn` — Yes (churned) / No (retained) |
| Churn Rate | ~26.5% — imbalanced dataset |

Features cover customer demographics, subscribed services (OnlineSecurity, StreamingTV, etc.), contract type, billing method, and monthly/total charges.

---

## Project Structure

```
telco-churn-supervised-learning/
│
├── CustomerChurn_SupervisedLearning.ipynb   # Fully executed notebook (Steps 1–8)
├── churn_model.pkl                          # Saved sklearn Pipeline (auto-created by Step 8)
├── summary_report.md                        # 400–500 word project summary
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## What's Inside the Notebook

| Step | What was done |
|---|---|
| **Step 1** | Theory notes — Customer Churn + CAC/CLV, Confusion Matrix business cost, Class Imbalance + SMOTE, 4 algorithm explanations, Precision vs Recall |
| **Step 2** | EDA on `df_eda = df.copy()` — TotalCharges dtype fix, class balance check, histograms (tenure, MonthlyCharges, TotalCharges), countplots (Contract, InternetService, PaymentMethod), churn rate by Contract, churn rate by tenure bucket, MonthlyCharges boxplot, correlation heatmap |
| **Step 3** | Preprocessing on `df_model = df.copy()` — drop customerID, fill TotalCharges median, engineer `tenure_group` / `num_services` / `AutoPay`, binary + OHE + label encoding, train-test split (80/20, stratified), StandardScaler, SMOTE on training set only |
| **Step 4** | KNN — baseline k=5 with all 5 metrics, k tuning (k = 1,3,5,7,9,11,15) + plot, best k confusion matrix + ROC curve |
| **Step 4** | Naive Bayes — all metrics, class prior probabilities check, confusion matrix + ROC, independence assumption note |
| **Step 5** | SVM — baseline (rbf, C=1) all metrics + confusion matrix, C tuning (0.1, 1, 10, 100) via 3-fold CV + plot, tuned metrics + ROC, black-box note |
| **Step 5** | Decision Tree — baseline (depth=5) metrics + confusion matrix, tree visualization (depth=3), depth tuning + plot, top 15 feature importances |
| **Step 5** | SMOTE vs `class_weight='balanced'` Recall comparison |
| **Step 6** | 4-model comparison table sorted by Recall, all 4 ROC curves on one figure, Precision vs Recall bar chart, deployment recommendation |
| **Step 7** | False Negative profiling vs overall churner profile, top 5 feature insights, 5 business-friendly churn signals |
| **Step 8** | sklearn Pipeline (StandardScaler + KNN), `joblib.dump` inside notebook, load + predict on 5 sample customers |

---

## Model Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **KNN** | 0.7154 | 0.4785 | **0.8021** | 0.5994 | 0.8121 |
| **Naive Bayes** | 0.7232 | 0.4869 | 0.7941 | 0.6037 | 0.8164 |
| **Decision Tree** | 0.7530 | 0.5240 | 0.7594 | **0.6201** | 0.8052 |
| **SVM** | 0.7239 | 0.4863 | 0.7112 | 0.5776 | 0.7884 |

> All metrics are for the **Churn class (class = 1)**, sorted by Recall (descending).
>
> **Primary metric: Recall** — because a missed churner (False Negative) costs the company the customer's full remaining CLV plus CAC to find a replacement.

---

## Key Churn Signals Found

1. **Month-to-Month contracts** — churn at 3–4x the rate of annual/two-year contract holders
2. **First 12 months (New tenure group)** — churn rate above 40%; the first year is the critical window
3. **High MonthlyCharges + no long-term contract** — most volatile, price-sensitive segment
4. **No AutoPay** — manual payers actively reconsider their bill each month
5. **Low num_services** — fewer add-ons = lower switching cost = easier to leave

---

## Recommended Model for Deployment

**KNN** achieved the highest Recall (0.80) — it catches the most actual churners. For production, **Decision Tree** is recommended as the deployment model because it balances Recall, Precision and F1-Score well while being fully interpretable. The retention team can read the tree rules directly and tailor the outreach message per customer (contract upgrade offer vs loyalty reward vs pricing plan review).

The saved **churn_model.pkl** (a sklearn Pipeline containing StandardScaler + KNN with best k) is generated automatically when Step 8 of the notebook is run. Load it with:

```python
import joblib
pipeline = joblib.load('churn_model.pkl')
predictions = pipeline.predict(new_customer_data)
probabilities = pipeline.predict_proba(new_customer_data)[:, 1]
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/telco-churn-supervised-learning.git
cd telco-churn-supervised-learning
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root folder.

**4. Run the notebook**
```bash
jupyter notebook CustomerChurn_SupervisedLearning.ipynb
```

Run all cells top to bottom. Step 8 will automatically create `churn_model.pkl` in the same folder.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (KNN, GaussianNB, SVC, DecisionTreeClassifier, StandardScaler, Pipeline) |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Model persistence | joblib |

---

## Author

**Herit Tanna**
Computer Engineering Diploma — A.V.P.T.I. Rajkot (GTU)
AI/ML & Data Science — Red & White Multimedia Education
