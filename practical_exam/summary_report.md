# Summary Report — Customer Churn Prediction

## Business Problem and Dataset Summary

This project addresses a critical business challenge for telecom companies like Jio or Airtel — predicting which customers are about to cancel their subscription so the retention team can intervene before it happens. The business case is built around two key concepts: CAC (Customer Acquisition Cost), which is the amount the company spends to bring in a new subscriber through marketing and onboarding, and CLV (Customer Lifetime Value), which is the total revenue a customer generates over their relationship with the company. Losing a customer permanently means losing their remaining CLV and spending a full CAC to replace them. A churn model reduces this loss by enabling early, targeted retention.

The dataset used is the IBM Telco Customer Churn benchmark with 7,043 customer records across 21 features. Features cover demographic information, service subscriptions (PhoneService, OnlineSecurity, StreamingTV, etc.), contract type (Month-to-Month, One Year, Two Year), billing details (PaymentMethod, PaperlessBilling), and monthly and total charges. The target variable is Churn (Yes = churned, No = retained). Approximately 26.5% of customers churned, confirming a moderate class imbalance that required handling before model training.

## Preprocessing and Class Imbalance Strategy

The TotalCharges column loaded as an object type because 11 customers with tenure = 0 had whitespace instead of a numeric value. These were converted to float and filled with the column median. Three new features were engineered from existing columns: tenure_group (binning raw tenure months into four lifecycle stages — New, Mid, Senior, Loyal), num_services (counting how many add-on services each customer subscribed to), and AutoPay (flagging customers on automatic payment methods, who tend to churn less). Binary columns were mapped to 0/1, multi-class categorical columns (InternetService, Contract, PaymentMethod) were one-hot encoded, and tenure_group was label encoded to preserve its natural ordinal order.

The data was split 80/20 with stratify=y to maintain the same churn ratio in both sets. StandardScaler was applied after the split (fitted only on training data) since KNN and SVM are distance-sensitive. SMOTE was applied exclusively to the training set to balance the churn classes from a 3:1 ratio to 1:1, generating synthetic churn examples by interpolating between real ones. The test set was left at its original imbalanced distribution to accurately represent real-world evaluation conditions.

## Which Model to Recommend and Why

| Model | Recall (Churn) | F1-Score | AUC-ROC |
|---|---|---|---|
| KNN | **0.8021** | 0.5994 | 0.8121 |
| Naive Bayes | 0.7941 | 0.6037 | 0.8164 |
| Decision Tree | 0.7594 | **0.6201** | 0.8052 |
| SVM | 0.7112 | 0.5776 | 0.7884 |

KNN achieved the highest Recall (0.80), meaning it caught the most actual churners in the test set. For production deployment, the **Decision Tree** is recommended because it delivers the best overall balance of Recall, Precision, F1-Score and interpretability. The retention team can read its decision rules directly, understand why a specific customer was flagged, and tailor the outreach message accordingly — something not possible with KNN or SVM. A saved sklearn Pipeline (StandardScaler + KNN) is included as churn_model.pkl for immediate inference use.

## Top 3 Churn Signals and Retention Actions

- **Month-to-Month contracts** are the single strongest churn predictor — these customers churn at 3–4x the rate of annual or two-year contract holders. The most impactful retention action is offering at-risk monthly customers a discounted contract upgrade (e.g., 20% off an annual plan).
- **Low tenure (first 12 months)** shows churn rates above 40%. The first year is the critical window. Proactive onboarding support, check-in calls at the 30-day and 90-day marks, and guided feature discovery during this period can significantly reduce early dropout.
- **High MonthlyCharges on a Month-to-Month contract** create the highest-risk combination. These customers pay premium prices with zero commitment — they have the strongest financial motivation and lowest switching friction. Personalized billing reviews or loyalty pricing for this segment can reduce price-driven churn.

## What Would You Do Next

The next steps for improving this system would be to test gradient boosting models (XGBoost, LightGBM) which typically outperform the classifiers used here on structured tabular data. Adding a real-time scoring API would allow the retention system to flag customers automatically as soon as their usage behaviour changes, rather than running batch predictions weekly. Collecting richer behavioural signals such as call centre contact history, network quality complaints, and competitor offer exposure would also significantly improve the model's ability to detect subtle early churn signals before they become final decisions to leave.
