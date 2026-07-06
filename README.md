<div align="center">

# 🛡️ RiskSentinel
### A Machine Learning Approach to Real-Time Fraud Scoring

*An end-to-end financial fraud detection system — from exploratory data analysis on 6.3M transactions to a real-time, deployable risk-scoring dashboard.*

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Overview

**RiskSentinel** is a machine learning system that flags potentially fraudulent financial transactions in real time. It's built on a full exploratory data analysis of a 6.3-million-row transaction dataset, a class-weighted classification pipeline, and a Streamlit dashboard for live scoring.

This README is also an honest account of the modeling process — including techniques that were **attempted but not completed** due to hardware limitations, and why the final deployed model looks the way it does.

| Approach | Status |
|---|---|
| **Logistic Regression, class-weighted** | ✅ Trained, evaluated, and deployed |
| **SMOTE oversampling** | 🧪 Implemented in code — training did not complete locally (RAM constraint) |
| **RandomizedSearchCV hyperparameter tuning** | 🧪 Implemented in code — not run to completion locally (RAM/compute constraint) |
| **XGBoost classifier** | 🧪 Attempted — not completed locally (RAM constraint) |

---

## 🎯 Problem Statement

This is a textbook **extreme class imbalance** problem:

- **Total transactions:** 6,362,620
- **Fraudulent transactions:** 8,213 (**≈ 0.13%** of all data)
- **Legitimate transactions:** 6,354,407 (**≈ 99.87%** of all data)

A model that always predicts "not fraud" would score ~99.8% accuracy while catching zero fraud. The real challenge is building something that can actually find that 0.13% — without generating so many false alarms that the model becomes unusable for a fraud analyst.

---

## 🔍 Exploratory Data Analysis

Key findings from EDA that shaped the modeling decisions:

- **Fraud only occurs in `TRANSFER` and `CASH_OUT` transaction types** — `PAYMENT` and `DEPOSIT` transactions had zero fraud cases in this dataset.
- A large number of transactions showed the sender's balance dropping to **exactly zero** after a `TRANSFER`/`CASH_OUT` — a strong behavioral fraud signal, visualized and counted separately during EDA.
- Transaction amount is heavily right-skewed (visualized on a log scale) — most transactions are small, fraud amounts vary widely.
- A correlation heatmap across `amount`, sender/receiver balances, and `isFraud` was used to sanity-check feature relationships before modeling.

**Engineered features:**
```python
df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
```
These capture the mismatch between the stated transaction amount and the actual change in account balances — one of the strongest fraud indicators in this type of data.

Non-predictive identifier columns (`nameOrig`, `nameDest`) and the business-rule flag (`isFlaggedFraud`) were dropped before modeling, since they don't generalize as learnable features.

---

## ⚙️ Modeling Approach

### ✅ Deployed Model — Logistic Regression (Class-Weighted)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                                'oldbalanceDest', 'newbalanceDest']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['type'])
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(X_train, y_train)
```

Instead of generating synthetic minority samples, class imbalance is handled **at the algorithm level** using `class_weight='balanced'` — the model penalizes misclassifying the rare fraud class more heavily during training, without needing to duplicate or synthesize any data. This keeps training memory-efficient on a 6.3M-row dataset with a standard `train_test_split` (70/30, stratified).

### 📈 Results (70/30 stratified split, test set = 1,908,786 transactions)

| Metric | Class 0 (Legitimate) | Class 1 (Fraud) |
|---|---|---|
| Precision | 1.00 | 0.02 |
| Recall | 0.95 | **0.94** |
| F1-Score | 0.97 | 0.04 |

**Overall accuracy:** 94.67% &nbsp;|&nbsp; **Confusion Matrix:**

| | Predicted: Legit | Predicted: Fraud |
|---|---|---|
| **Actual: Legit** | 1,804,747 | 101,575 |
| **Actual: Fraud** | 151 | 2,313 |

**How to read this:** the model catches **94% of all actual fraud** (recall) — out of 2,464 real fraud cases in the test set, it correctly flags 2,313 and misses only 151. The trade-off is precision: only ~2% of transactions flagged as "fraud" are actually fraud, meaning a large number of legitimate transactions get flagged for review.

This is a deliberate, common trade-off in fraud detection: **missing real fraud (false negatives) is usually far more costly than flagging a legitimate transaction for manual review (false positives)**. The current model is tuned toward high recall for exactly this reason — but the precision gap is the clearest next target for improvement (see below).

---

## 🧪 Attempted Approaches (Not Completed — Documented for Transparency)

The following techniques were implemented in the notebook to improve on the baseline, but **could not be run to completion on local hardware** due to the dataset's size (6.3M rows). Rather than leave them out, they're documented here as-is, along with why they matter and how they'd typically be resolved:

**1. SMOTE (Synthetic Minority Oversampling)**
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipeline = Pipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000))
])
```
SMOTE generates synthetic fraud samples by interpolating between existing minority-class points. On a dataset this size, the nearest-neighbor search and sample generation step became too memory-intensive for the available local RAM.

**2. RandomizedSearchCV (Hyperparameter Tuning)**
```python
from sklearn.model_selection import RandomizedSearchCV
```
Imported and set up to tune the classifier via cross-validated random search, but running multiple CV folds across 6.3M rows was not feasible without more memory/compute than was locally available.

**3. XGBoost**
An XGBoost classifier was explored as a stronger alternative to Logistic Regression (tree-based models generally handle imbalanced tabular data better and offer a built-in `scale_pos_weight` parameter as a lighter-weight alternative to oversampling). Training could not be completed locally at full data scale.

**Planned path forward for all three** (see [Roadmap](#-roadmap)):
- Train on a stratified subsample before scaling up
- Move training to a higher-memory cloud environment (AWS/GCP/Azure, or Colab Pro)
- Use `scale_pos_weight` in XGBoost/LightGBM instead of SMOTE, which avoids synthetic data generation entirely
- Use `RandomUnderSampler` as a memory-light alternative to SMOTE

---

## 🖥️ Application

The final model is deployed as an interactive **Streamlit** dashboard:

- 💳 Manual transaction input with real-time feature engineering
- 📊 Plotly-based fraud risk gauge for visual risk scoring
- 🧪 One-click sample transactions (fraud & legitimate) for quick demos
- 🕓 Session-based prediction history
- 📄 Downloadable per-transaction fraud report

---

## 🧰 Tech Stack

- **Language:** Python 3.10
- **ML:** scikit-learn, imbalanced-learn
- **Data Handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn (EDA), Plotly (dashboard)
- **Deployment/UI:** Streamlit
- **Model Serialization:** joblib

---

## 📁 Project Structure

```
RiskSentinel/
│
├── app.py                          # Streamlit dashboard (deployed app)
├── Fraud_Detection.ipynb           # EDA + model training notebook
├── Fraud_Detection_pipeline.pkl    # Trained, deployed model (Logistic Regression)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ChankumarSah/RiskSentinel.git
cd RiskSentinel

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🗺️ Roadmap

- [ ] Complete SMOTE training on a stratified subsample or higher-memory environment
- [ ] Run `RandomizedSearchCV` for hyperparameter tuning at scale
- [ ] Train and benchmark an XGBoost model with `scale_pos_weight`
- [ ] Improve precision on the fraud class without sacrificing current recall
- [ ] Add SHAP-based model explainability for individual predictions
- [ ] Add batch (CSV upload) prediction mode
- [ ] Deploy a live public demo

---

## 👨‍💻 Author

**Chandan Kumar Sah**
Machine Learning & Data Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/chandan-kumar-sah-752803387/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ChankumarSah)

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and build on it.

<div align="center">

*If this project was useful or interesting, consider leaving a ⭐ on the repo.*

</div>
