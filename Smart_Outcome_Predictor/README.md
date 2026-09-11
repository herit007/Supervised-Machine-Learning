# Smart Outcome Predictor
### Ensemble-Based Machine Learning System for Student Outcome Prediction

> Predicting whether a student will complete an online course and what final score they will achieve — using Bagging, Boosting, Voting and Stacking ensemble methods.

---

## Business Problem

An EdTech analytics company wants to predict student outcomes early so instructors and platform teams can intervene before a student drops out or underperforms. Two prediction tasks are required:

- **Classification Task:** Will the student successfully complete the course? (`completion_status` — 0 = Not Completed, 1 = Completed)
- **Regression Task:** What final performance score will the student achieve? (`final_score` — continuous, range 35–100)

Single models produce unstable results. This project designs an **ensemble-based solution** using Bagging, Boosting (AdaBoost, Gradient Boosting, LightGBM, XGBoost), Voting (Hard + Soft), and Stacking to improve accuracy and robustness across both tasks.

---

## Dataset

| Property | Detail |
|---|---|
| File | `Smart_Outcome_Predictor.csv` |
| Rows | 5,200 student records |
| Features | 19 columns |
| Classification Target | `completion_status` (0 = Not Completed, 1 = Completed) |
| Regression Target | `final_score` (0–100) |
| Completion Rate | ~37.5% — moderate class imbalance |

Features cover student demographics (age, country_region, device_type, education_background), course details (course_level, course_category), and engagement metrics (sessions, time_spent_hours, videos_watched, quiz_attempts, assignments_submitted, avg_quiz_score, attendance_rate).

---

## Project Structure

```
smart-outcome-predictor/
│
├── Smart_Outcome_Predictor.ipynb   # Fully executed notebook (Parts A–G)
├── PART_A_Theory.pdf               # Theory answers (Ensemble Learning concepts)
└── README.md                       # This file
```

---

## What's Inside the Notebook

| Part | What was done |
|---|---|
| **Part A** | Theory as markdown cells — Ensemble Learning Methods, Bagging vs Boosting vs Voting, Bias-Variance Trade-off, Voting Classifier + Stacking, AdaBoost vs GradientBoosting vs LightGBM vs XGBoost |
| **Part B** | EDA on `df_eda = df.copy()` — distributions, completion rate by course level, final score by category, correlation heatmap. Preprocessing on `df_model = df.copy()` — drop student_id + date, fill missing medians, label encode categoricals, identify features + targets, train-test split (80/20), StandardScaler |
| **Part C** | Single Decision Tree baseline (classification + regression), BaggingClassifier (50 trees), BaggingRegressor (50 trees), direct comparison vs single base model |
| **Part D** | AdaBoost Classifier + Regressor + staged_score plot showing sequential improvement. Gradient Boosting Classifier + Regressor + learning rate / n_estimators analysis. LightGBM Classifier + Regressor + training time efficiency comparison vs GB. XGBoost Classifier + Regressor + boosting model comparison table |
| **Part E** | Hard Voting Classifier (DT + LR + KNN), Soft Voting Classifier, Hard vs Soft comparison. Stacking Classifier (DT + LightGBM + KNN → Logistic Regression meta-learner). Stacking Regressor (DT + LightGBM + GB → Linear Regression meta-learner) |
| **Part F** | Full classification comparison table (Accuracy, Precision, Recall, F1, AUC-ROC), full regression comparison table (MAE, RMSE, R²), ROC curve plot for all classifiers, RMSE bar chart, best model identified for each task |
| **Part G** | Final report — bagging vs boosting impact, boosting algorithm comparison, voting/stacking advantages, deployment recommendation for both tasks |

---

## Classification Results (Test Set — Sorted by F1-Score)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **XGBoost Classifier** | **0.7317** | 0.6637 | 0.5769 | **0.6173** | **0.7850** |
| Gradient Boosting Classifier | 0.7337 | 0.6707 | 0.5692 | 0.6158 | 0.7872 |
| Stacking Classifier | 0.7346 | 0.6781 | 0.5564 | 0.6113 | 0.7822 |
| AdaBoost Classifier | 0.7365 | 0.6883 | 0.5436 | 0.6074 | 0.7836 |
| LightGBM Classifier | 0.7250 | 0.6538 | 0.5667 | 0.6071 | 0.7718 |
| Soft Voting | 0.7231 | 0.6667 | 0.5231 | 0.5862 | 0.7735 |
| Hard Voting | 0.7288 | 0.6888 | 0.5051 | 0.5828 | 0.7735 |
| Bagging Classifier | 0.7058 | 0.6243 | 0.5410 | 0.5797 | 0.7581 |
| Single Decision Tree | 0.6231 | 0.4976 | 0.5359 | 0.5160 | 0.6056 |

---

## Regression Results (Test Set — Sorted by RMSE)

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Gradient Boosting Regressor** | **7.8718** | **9.7958** | **0.4864** |
| Stacking Regressor | 7.8689 | 9.8344 | 0.4824 |
| XGBoost Regressor | 7.8960 | 9.8366 | 0.4821 |
| LightGBM Regressor | 7.9574 | 9.8919 | 0.4763 |
| Bagging Regressor | 8.0772 | 10.0908 | 0.4550 |
| AdaBoost Regressor | 8.5829 | 10.5423 | 0.4052 |
| Single Decision Tree | 11.0087 | 13.8751 | -0.0304 |

> Single Decision Tree (R² = -0.03) performed worse than a flat mean prediction, confirming exactly why ensemble methods are needed for this problem.

---

## Key Findings

**Bagging vs Boosting:**
Bagging improved significantly over the single Decision Tree by reducing variance — accuracy jumped from 0.62 to 0.71 for classification. Boosting methods (especially XGBoost and Gradient Boosting) went further by also reducing bias, pushing F1-Score above 0.61 and AUC-ROC above 0.78.

**Best Boosting Algorithm:**
LightGBM and XGBoost were the strongest and most efficient boosting approaches. XGBoost had the best F1-Score (0.6173) for classification. Gradient Boosting Regressor gave the lowest RMSE (9.7958) for regression. LightGBM trained significantly faster than standard Gradient Boosting with near-identical accuracy.

**Voting and Stacking:**
Soft Voting outperformed Hard Voting because it uses full probability distributions rather than majority labels. Stacking Classifier outperformed Soft Voting in precision and overall accuracy by using a meta-learner (Logistic Regression) that learned the optimal combination of base model outputs.

---

## Recommended Models for Deployment

| Task | Recommended Model | Reason |
|---|---|---|
| Classification (completion_status) | **XGBoost Classifier** | Best F1-Score (0.6173) and strong AUC-ROC (0.7850) with built-in regularization |
| Regression (final_score) | **Gradient Boosting Regressor** | Lowest RMSE (9.7958) and highest R² (0.4864) |

For a production EdTech platform where interpretability matters (showing instructors *why* a student is at risk), **LightGBM** is also recommended since it provides feature importances, trains much faster, and is easy to retrain as new cohort data arrives monthly.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/smart-outcome-predictor.git
cd smart-outcome-predictor
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
```

**3. Add the dataset**

Place `Smart_Outcome_Predictor.csv` in the project root folder.

**4. Run the notebook**
```bash
jupyter notebook Smart_Outcome_Predictor.ipynb
```

Run all cells from top to bottom.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (Bagging, AdaBoost, GradientBoosting, VotingClassifier, StackingClassifier, StackingRegressor, Pipeline) |
| Boosting | lightgbm, xgboost |

---

## Author

**Herit Tanna**
Computer Engineering Diploma — A.V.P.T.I. Rajkot (GTU)
AI/ML & Data Science — Red & White Multimedia Education
