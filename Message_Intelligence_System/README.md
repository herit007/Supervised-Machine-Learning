# 📩 Message Intelligence System

A machine learning based classification system that automatically detects whether an incoming message is **Spam** or **Legitimate**, built using probability, distance-based, and margin-based classifiers.

---

## 🔗 Quick Links

| Resource | Location |
|---|---|
| 📓 Jupyter Notebook | [`Notebook/Message_Intelligence_System.ipynb`](./Notebook/Message_Intelligence_System.ipynb) |
| 📊 Dataset | [`Dataset/Message_Intelligence_Dataset.csv`](./Dataset/Message_Intelligence_Dataset.csv) |
| 📄 Theory / Concepts (PDF) | [`Theory.pdf`](./Theory.pdf) |
| 🎥 Video Explanation | [`Explanation Video`](#) |

---

## 📌 Project Overview

**Message Intelligence System** is a data science project built for a communication security company scenario, where the goal is to automatically flag messages as:

- `0` → Legitimate Message
- `1` → Spam Message

The project combines **probability theory** with **distance-based**, **margin-based**, and **probabilistic** classifiers, and studies how different modeling assumptions affect classifier performance.

---

## 🎯 Objective

To design, build, and compare multiple classification models — **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Naive Bayes** — that can accurately classify messages as spam or legitimate, and to explain their working using core probability concepts.

---

## 🧠 What This Project Does

- Performs Exploratory Data Analysis (EDA) on message-related features
- Handles missing values and scales data correctly (fit only on training data, avoiding data leakage)
- Extracts useful information from a timestamp column (month, weekend indicator)
- Implements and tunes a **KNN** classifier, experimenting with different values of K and distance metrics
- Implements an **SVM** classifier with Linear and RBF kernels, and analyzes support vectors
- Implements a **Naive Bayes** classifier, including a manual, from-scratch demonstration of conditional probability and Bayes' Theorem
- Evaluates and compares all three models using Accuracy, Precision, Recall, and F1 Score
- Recommends the best model for real-world deployment

---

## 🗂️ Dataset

The dataset contains message-related features extracted from text and user behavior signals, including:

- Message length, word count
- Number of special characters, digits, and URLs
- Spam and legitimate keyword scores
- Sender activity score, account age, and recent activity
- Time-based signals (hour of day, day of week)
- **Target column:** `spam_label`

---

## 🛠️ Tech Stack & Libraries

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (KNN, SVM, Naive Bayes) |
| Environment | Jupyter Notebook |

---

## ⚙️ Project Workflow

1. **Data Understanding & Preparation** – Explore the dataset, handle missing values, extract features from timestamp, split into train/test, and scale correctly
2. **Baseline Model (KNN)** – Implement, tune K, compare distance metrics, and analyze misclassifications
3. **Support Vector Machine** – Implement Linear and RBF kernels, study support vectors and margins
4. **Naive Bayes & Probability** – Implement the classifier and manually demonstrate Bayes' Theorem
5. **Model Comparison & Evaluation** – Compare all models on Accuracy, Precision, Recall, and F1 Score
6. **Final Analysis & Reporting** – Summarize strengths, weaknesses, trade-offs, and give a business recommendation

---

## 📊 Results

All three models were evaluated on the same test set using Accuracy, Precision, Recall, and F1 Score, with results and comparison charts available in the notebook. The final recommended model is chosen based on the best overall F1 Score, balancing false spam flags against missed spam messages.

---

## 🚀 How to Run

```bash
git clone <this-repo-url>
cd Message-Intelligence-System
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Notebook/Message_Intelligence_System.ipynb
```

---

## 📁 Folder Structure

```
Message_Intelligence_System/
│
├── Dataset/
│   └── Message_Intelligence_Dataset.csv
│
├── Notebook/
│   └── Message_Intelligence_System.ipynb
│
├── Theory.pdf
│
└── README.md
```

---

## 👤 Author

**Herit**

---

⭐ If you found this project useful or interesting, consider giving it a star!
