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
- **Handled Missing Values**: Imputed missing numerical values and dropped non-informative columns.  
- **Removed Duplicates**: Ensured unique transactions.  
- **Corrected Data Types**: Converted time columns and enforced proper numerical/categorical types.  

### 2. Exploratory Data Analysis (EDA)
- Distribution plots of features such as `age`, `purchase_value`, and fraud classes.  
- Relationship analysis between fraud rate and behavioral/geographical features.  

### 3. Geolocation Analysis
- Converted IP addresses to integers.  
- Mapped IP ranges to countries using interval logic.  

### 4. Feature Engineering
- Transaction frequency by user.  
- Time-based features (`hour_of_day`, `day_of_week`, `time_since_signup`).  

### 5. Data Transformation
- **SMOTE** applied to training set for class balance.  
- **Scaling** numerical features with `StandardScaler`.  
- **One-Hot Encoding** categorical variables such as `browser`, `source`, `sex`, and `country`.  

---

## Unit Testing

Unit tests were added to improve **reliability** and **reproducibility**:
- Tested visualization functions to ensure plots are generated without errors.  
- Tested IP-to-country mapping logic for correct geolocation assignment.  
- Automated tests allow consistent validation when code changes are made.  

##  Final Output

The final dataset was:
- Clean and structured
- Feature-enriched
- Class-balanced
- Scaled and encoded
- Ready for model training and evaluation

---

##  Tools Used

- `pandas`, `numpy` – for data manipulation
- `matplotlib`, `seaborn` – for data visualization
- `scikit-learn` – for preprocessing, modeling, and evaluation
- `imblearn` – for handling class imbalance with SMOTE

---
