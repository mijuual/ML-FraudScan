#  Fraud Detection Project

##  Project Overview

Online fraud is a critical issue for businesses that rely on digital transactions. Fraudulent purchases can lead to substantial financial losses, damage to brand reputation, and degraded customer trust. The business need for this project is to proactively detect and prevent fraudulent activity based on user behavior, device/browser information, and geolocation data.

The objective of this assignment is to analyze historical transaction data and build a robust fraud detection pipeline by:

- Preprocessing and cleaning the raw data.
- Enriching it with engineered features (e.g., time-based metrics, geolocation).
- Handling class imbalance through sampling techniques.
- Preparing the dataset for predictive modeling by encoding, scaling, and transforming features.

---

## Methodolgy

### 1. Data Cleaning & Preprocessing

- **Handled Missing Values**:
  - Inspected missing data.
  - Imputed missing numerical values using the mean strategy.
  - Dropped non-informative or fully missing columns (e.g., `time_since_last_txn`).

- **Removed Duplicates**:
  - Checked and dropped duplicate rows from the dataset to maintain data integrity.

- **Corrected Data Types**:
  - Converted `signup_time` and `purchase_time` to `datetime64`.
  - Ensured numerical and categorical columns had appropriate data types.

---

### 2.  Exploratory Data Analysis (EDA)

- **Univariate Analysis**:
  - Plotted distributions of features such as `age`, `purchase_value`, and class balance.

- **Bivariate Analysis**:
  - Investigated relationships between variables (e.g., fraud rate vs. age, fraud rate vs. purchase value).
  - Used plots and `groupby` analysis to understand fraud patterns.

---

### 3.  Geolocation Analysis

- **IP Address Conversion**:
  - Converted `ip_address` from float format to integer.

- **IP Mapping**:
  - Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` by matching `ip_address` with IP ranges (`lower_bound_ip_address`, `upper_bound_ip_address`).
  - Enriched the dataset with a `country` column for geographical fraud analysis.

---

### 4.  Feature Engineering

- **User Behavior Features**:
  - Calculated transaction frequency using `user_id`.

- **Time-Based Features**:
  - `hour_of_day`: Extracted from `purchase_time`.
  - `day_of_week`: Extracted from `purchase_time`.
  - `time_since_signup`: Time (in hours) between `signup_time` and `purchase_time`.

---

### 5.  Data Transformation

####  Class Imbalance Handling

- Identified severe class imbalance (~9% fraud).
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to the **training set only** to balance the classes.
- After SMOTE, the training data was evenly split (50/50) between fraud and non-fraud.

> **Justification**: SMOTE was chosen for its ability to synthetically generate realistic examples of the minority class, improving model performance on fraud cases.

---

####  Normalization and Scaling

- Applied `StandardScaler` to normalize numerical features such as `purchase_value`, `age`, and `time_since_signup`.
- Prevented data leakage by fitting the scaler only on the training set.

---

####  Categorical Feature Encoding

- Used **One-Hot Encoding** on categorical features like `browser`, `device_id`, `source`, `sex`, and `country`.
- Set `handle_unknown='ignore'` to prevent errors during transformation on the test set.

---

##  Final Output

The final dataset was:
- Clean and structured
- Feature-enriched
- Class-balanced
- Scaled and encoded
- Ready for model training and evaluation

---

##  Next Steps

With the dataset prepared, it can be moved on to model training using:
- Logistic Regression
- Random Forest
- XGBoost
- Neural Networks
- Or other advanced classifiers

---

##  Tools Used

- `pandas`, `numpy` – for data manipulation
- `matplotlib`, `seaborn` – for data visualization
- `scikit-learn` – for preprocessing, modeling, and evaluation
- `imblearn` – for handling class imbalance with SMOTE

---
