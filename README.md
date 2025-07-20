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

#### 1. Handle Missing Values
- Checked all datasets for missing values.
- Dropped or imputed missing values where appropriate.

#### 2. Data Cleaning
- Removed duplicate rows.
- Ensured correct data types (e.g., datetime parsing for `signup_time` and `purchase_time`).
- Converted IP addresses to integer format for merging.

#### 3. Exploratory Data Analysis (EDA)
- Conducted univariate analysis (e.g., class imbalance, feature distributions).
- Performed bivariate analysis (e.g., fraud by country, fraud by time of day).

#### 4. Merge Datasets for Geolocation
- Converted IP address ranges in `IpAddress_to_Country.csv` to numeric format.
- Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` based on IP ranges to add location data.

#### 5. Feature Engineering
- Extracted **hour of day** and **day of week** from `purchase_time`.
- Calculated `time_since_signup` (difference between `purchase_time` and `signup_time`).
- Engineered **transaction frequency and velocity features** to capture user behavior patterns.

#### 6. Data Transformation

To prepare the data for modeling, I performed the following steps:

- **Train-Test Split**: Split the dataset into 80% training and 20% testing sets using stratified sampling to preserve class distribution.

- **Column Identification**:
  - Categorical columns: Identified by data type (`object`, `category`).
  - Numerical columns: Identified by data type (`int64`, `float64`).

- **Preprocessing Pipelines**:
  - **Numerical** features were scaled using `StandardScaler`.
  - **Categorical** features were encoded using `OneHotEncoder` with `handle_unknown='ignore'`.

- **Class Imbalance Handling**:
  - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the **training set only**, after preprocessing.
  - This generated synthetic examples for the minority fraud class, balancing the dataset.

- **Final Outputs**:
  - `X_train_resampled`, `y_train_resampled`: Fully preprocessed and balanced training set.
  - `X_test_processed`, `y_test`: Preprocessed test set.