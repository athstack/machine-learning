# 📊 Retail Customer Churn Prediction

End-to-end machine learning pipeline for predicting customer churn in a retail dataset.

## 📓 Interactive Notebook

View the complete Jupyter Notebook with all analysis, visualizations, diagrams, model evaluation results, and outputs:

[![View Notebook](https://img.shields.io/badge/📓%20View%20Notebook-NBViewer-orange?style=for-the-badge)](https://nbviewer.org/github/athstack/customer-prediction/blob/main/Assignment4_Retail_ML_FINAL2.ipynb)

---

## 📁 Project Structure

```text
Retail_Customer/
│
├── data/
│   ├── customers.csv
│   │   # 5,000 customers
│   │
│   ├── transactions.csv
│   │   # 32,295 transactions
│   │
│   ├── interactions.csv
│   │   # 100,000 interactions
│   │
│   ├── support_tickets.csv
│   │   # 3,000 tickets
│   │
│   ├── customer_reviews_complete.csv
│   │   # 1,108 reviews
│   │
│   ├── campaigns.csv
│   │   # 200 campaigns
│   │
│   └── processed/
│       └── final_ml_dataset.csv
│           # Processed ML dataset
│
├── models/
│   └── retail_churn_pipeline.joblib
│       # Serialized machine learning pipeline
│
├── Assignment4_Retail_ML.ipynb
│   # Original notebook
│
├── Assignment4_Retail_ML_CORRECTED.ipynb
│   # Corrected notebook
│
├── Assignment4_Retail_ML_FINAL2.ipynb
│   # Final notebook
│
├── app.py
│   # Streamlit deployment application
│
└── requirements.txt
    # Python dependencies
```

---

## 🎯 Project Objective

The objective of this project is to develop an end-to-end machine learning pipeline for predicting whether a retail customer is likely to churn.

The project integrates multiple retail datasets and applies data analysis, feature engineering, machine learning, model evaluation, and deployment techniques.

The complete workflow is:

```text
Raw Datasets
     ↓
Data Inspection
     ↓
Data Validation
     ↓
Primary / Foreign Key Analysis
     ↓
Relationship Identification
     ↓
LEFT JOIN Data Integration
     ↓
Merge Validation
     ↓
Temporal Cutoff
     ↓
Feature Engineering
     ↓
Churn Target Creation
     ↓
Preprocessing
     ↓
Feature Selection
     ↓
Model Training
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Error Analysis
     ↓
Model Serialization
     ↓
Streamlit Deployment
```

---

## 📋 Assignment Requirements

The project addresses the following assignment requirements:

1. Load and inspect all datasets
2. Check primary key uniqueness
3. Identify parent-child relationships
4. Determine the appropriate `LEFT JOIN` merge strategy
5. Validate row counts before and after merges
6. Detect unmatched keys
7. Document missing values introduced by JOIN operations
8. Perform feature engineering:
   - RFM
   - Engagement
   - Support
   - Review features
9. Apply a temporal cutoff to prevent data leakage
10. Perform a temporal leakage audit on support tickets
11. Create the churn target variable
12. Build a preprocessing pipeline:
   - Imputation
   - Encoding
   - Scaling
13. Perform an 80/20 stratified train/test split
14. Perform feature selection using `SelectKBest`
15. Evaluate multiple feature-selection `k` values
16. Train four machine learning models
17. Perform 5-Fold Stratified Cross-Validation and `GridSearchCV`
18. Evaluate models using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
19. Perform False Positive / False Negative error analysis
20. Deploy the trained model using Joblib and Streamlit

---

## 🔍 Data Inspection

All datasets are inspected independently before integration.

The inspection process includes:

- Dataset dimensions
- Column names
- Data types
- Sample records
- Missing values
- Duplicate records
- Statistical summaries
- Primary key uniqueness

---

## 🔑 Primary Keys and Relationships

The project identifies primary keys and parent-child relationships before performing any merge operations.

The general relationship structure is:

```text
Customers
    │
    ├── Transactions
    │
    ├── Interactions
    │
    ├── Support Tickets
    │
    └── Customer Reviews
```

This relationship analysis ensures that the correct join keys and merge strategy are used.

---

## 🔗 LEFT JOIN Merge Strategy

A `LEFT JOIN` strategy is used to preserve the customer population while incorporating information from related datasets.

```text
Customers
    │
    ├── LEFT JOIN → Transactions
    │
    ├── LEFT JOIN → Interactions
    │
    ├── LEFT JOIN → Support Tickets
    │
    └── LEFT JOIN → Customer Reviews
```

The `LEFT JOIN` approach ensures that customers without corresponding records in child datasets remain in the final customer-level dataset.

---

## 📏 Merge Validation

Row counts are recorded before and after every merge.

This validation helps identify:

- Incorrect join keys
- Duplicate keys
- Unexpected row multiplication
- Unexpected record loss
- Referential integrity problems

---

## 🔍 Unmatched Keys and Missing Values

Unmatched keys are detected and documented after the merge operations.

The analysis identifies customers who may not have:

- Transactions
- Interactions
- Support tickets
- Reviews

Missing values introduced by `LEFT JOIN` operations are handled during the preprocessing stage.

---

## 🧮 Feature Engineering

The project creates customer-level features from the different datasets.

### RFM Features

RFM analysis consists of:

- **Recency** — how recently a customer purchased
- **Frequency** — how frequently a customer purchased
- **Monetary** — how much a customer spent

### Engagement Features

Customer interaction data is used to create engagement-related features.

### Support Features

Support-ticket information is transformed into customer-level support features.

### Review Features

Customer reviews are used to create review-related features such as review activity and ratings.

---

## ⏳ Temporal Cutoff and Data Leakage Prevention

A temporal cutoff is applied to prevent future information from being used during model development.

```text
Historical Customer Data
          ↓
   Temporal Cutoff
          ↓
Information Available
Before Prediction Time
          ↓
   Feature Engineering
          ↓
    Model Training
```

This ensures that the model only uses information that would realistically have been available at prediction time.

---

## 🔎 Temporal Leakage Audit

A temporal leakage audit is performed on support-ticket data.

The audit verifies that support tickets occurring after the prediction cutoff are not incorrectly included in the feature calculations.

---

## 🎯 Churn Target Variable

The target variable represents whether a customer churned.

```text
Churn
├── 0 → Customer did not churn
└── 1 → Customer churned
```

This makes the problem a supervised binary classification task.

---

## ⚙️ Preprocessing Pipeline

The machine learning pipeline performs:

```text
Raw Features
      ↓
Missing Value Imputation
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
Feature Selection
      ↓
Machine Learning Model
```

Using a pipeline ensures that the same transformations are applied consistently during training and prediction.

---

## ✂️ Feature Selection

Feature selection is performed using:

```python
SelectKBest
```

Multiple `k` values are evaluated to determine an appropriate number of predictive features.

---

## 🤖 Machine Learning Models

Four machine learning models are trained and compared.

The models are evaluated using the same training, validation, and testing strategy to provide a fair comparison.

---

## 🔬 Cross-Validation and Hyperparameter Tuning

The project uses **5-Fold Stratified Cross-Validation** to maintain the class distribution across validation folds.

Hyperparameter optimization is performed using:

```python
GridSearchCV
```

This allows different hyperparameter combinations to be evaluated systematically.

---

## 📈 Model Evaluation

The following metrics are used:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctness of positive churn predictions |
| Recall | Ability to identify actual churners |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between churn and non-churn |

---

## 🔎 Error Analysis

The project analyzes prediction errors using False Positives and False Negatives.

### False Positive

```text
Actual:    No Churn
Predicted: Churn
```

### False Negative

```text
Actual:    Churn
Predicted: No Churn
```

This analysis helps identify where the model performs poorly and which type of error requires more attention.

---

## 💾 Model Serialization

The final machine learning pipeline is saved using Joblib:

```text
models/
└── retail_churn_pipeline.joblib
```

The serialized pipeline can be loaded later without retraining the model.

---

## 🌐 Streamlit Deployment

The trained model is deployed through a Streamlit application.

Application file:

```text
app.py
```

Run the application with:

```bash
streamlit run app.py
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/athstack/customer-prediction.git
```

### 2. Navigate to the Project

```bash
cd customer-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Final Notebook

```bash
jupyter notebook Assignment4_Retail_ML_FINAL2.ipynb
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 🔄 Data Lineage

The complete data lineage is:

```text
Raw CSV Files
      ↓
Date Conversion
      ↓
Temporal Cutoff
      ↓
Feature Engineering
      ↓
LEFT JOIN
      ↓
Merge Validation
      ↓
Missing Value Handling
      ↓
Preprocessing
      ↓
Feature Selection
      ↓
Model Training
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Error Analysis
      ↓
Joblib Serialization
      ↓
Streamlit Deployment
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **SelectKBest**
- **GridSearchCV**
- **Joblib**
- **Jupyter Notebook**
- **Streamlit**

---

## 📦 Dependencies

All Python dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

- Data cleaning
- Data validation
- Relational data integration
- Primary and foreign keys
- LEFT JOIN operations
- Missing-value handling
- Exploratory Data Analysis
- RFM analysis
- Feature engineering
- Temporal data analysis
- Data leakage prevention
- Binary classification
- Feature selection
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Error analysis
- Model serialization
- Machine learning deployment

---

## 📊 Dataset Summary

| Dataset | Records | Purpose |
|---|---:|---|
| `customers.csv` | 5,000 | Customer information |
| `transactions.csv` | 32,295 | Customer transactions |
| `interactions.csv` | 100,000 | Customer interactions |
| `support_tickets.csv` | 3,000 | Support activity |
| `customer_reviews_complete.csv` | 1,108 | Customer reviews |
| `campaigns.csv` | 200 | Marketing campaigns |
| `final_ml_dataset.csv` | Processed | Final ML dataset |

---

## 👨‍💻 Author

### athstack

**Aspiring Data Scientist | Data Analytics | Machine Learning | Software Development**

GitHub: [@athstack](https://github.com/athstack)

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📓 Explore the Complete Analysis

[![Open Interactive Notebook](https://img.shields.io/badge/📓%20OPEN%20INTERACTIVE%20NOTEBOOK-NBViewer-orange?style=for-the-badge)](https://nbviewer.org/github/athstack/customer-prediction/blob/main/Assignment4_Retail_ML_FINAL2.ipynb)
