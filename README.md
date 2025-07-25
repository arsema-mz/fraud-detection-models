# Fraud Detection Project

This project aims to improve fraud detection for both e-commerce and bank credit card transactions. It explores machine learning models, class imbalance handling, and explainability tools.

## Datasets
- Fraud_Data.csv
- IpAddress_to_Country.csv
- creditcard.csv


## 🧠 Data Analysis and Preprocessing

This task involved preparing the data for machine learning by performing cleaning, feature engineering, and merging across three datasets:

- `Fraud_Data.csv`: Contains user and transaction-related fraud information.
- `IpAddress_to_Country.csv`: Maps IP addresses to countries.
- `creditcard.csv`: Contains anonymized credit card transactions for additional fraud detection.

### ✅ Steps Completed

Data preparation involved cleaning, feature engineering, and merging across the three datasets to create a robust input for modeling:

- **Handled Missing Values**: Checked and addressed missing values through dropping or imputation.
- **Data Cleaning**: Removed duplicates, ensured correct data types, and converted IP addresses for merging.
- **Exploratory Data Analysis (EDA)**: Analyzed feature distributions and class imbalance.
- **Merged Datasets**: Combined datasets to enrich transaction data with geolocation information.
- **Feature Engineering**: Extracted temporal features and engineered transaction frequency and velocity metrics.
- **Data Transformation**: Performed a train-test split, scaled numerical features, encoded categorical variables, and addressed class imbalance using SMOTE on the training set.


## ⚙️ Model Training and Evaluation

Two machine learning models were trained and evaluated for fraud detection:

### 1. Logistic Regression
- **Confusion Matrix**:
  - True Negatives (TN): 26,920
  - False Positives (FP): 473
  - False Negatives (FN): 1,243
  - True Positives (TP): 1,587
- **ROC AUC**: 0.84
- **Precision-Recall AUC**: 0.65

### 2. Random Forest
- **Confusion Matrix**:
  - True Negatives (TN): 27,263
  - False Positives (FP): 130
  - False Negatives (FN): 1,326
  - True Positives (TP): 1,504
- **ROC AUC**: 0.84
- **Precision-Recall AUC**: 0.70

### Summary of Findings
- Both models achieved a similar ROC AUC of 0.84, indicating good performance in class differentiation.
- Random Forest outperformed Logistic Regression in precision and in handling false classifications.
