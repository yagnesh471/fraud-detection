# Credit Card Fraud Detection using Machine Learning

A machine learning project for detecting fraudulent credit card transactions using XGBoost and SMOTE.

The project covers the complete machine learning workflow, including data exploration, preprocessing, class imbalance handling, model training, evaluation, model persistence, and transaction-level fraud prediction.

---

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions are extremely rare compared to legitimate transactions.

This project builds a fraud detection system that:

- Loads and explores transaction data
- Separates features and target
- Splits data into training and testing sets
- Scales the features
- Handles class imbalance using SMOTE
- Trains an XGBoost classifier
- Evaluates the model using multiple metrics
- Saves the trained model and scaler
- Predicts whether a transaction is legitimate or fraudulent

---

## Dataset

The project uses the Credit Card Fraud Detection dataset.

The dataset contains:

- 284,807 total transactions
- 284,315 legitimate transactions
- 492 fraudulent transactions

### Class Distribution

| Class | Transactions | Percentage |
|---|---:|---:|
| Legitimate (0) | 284,315 | 99.827% |
| Fraud (1) | 492 | 0.173% |

The original dataset is not included in this GitHub repository because it is approximately 150 MB.

The dataset should be placed locally at:

```text
data/creditcard.csv




    Features

The dataset contains 30 input features:

Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount

The target variable is:

Class

where:

0 → Legitimate
1 → Fraud
Machine Learning Pipeline
Credit Card Dataset
        ↓
Data Exploration
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
SMOTE on Training Data
        ↓
XGBoost Model
        ↓
Model Evaluation
        ↓
Save Model + Scaler
        ↓
Fraud Prediction
Handling Class Imbalance

The dataset contains far fewer fraudulent transactions than legitimate transactions.

The training set originally contained:

Class 0 → 227,451
Class 1 → 394

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training data.

After SMOTE:

Class 0 → 227,451
Class 1 → 227,451

Resulting training dataset:

454,902 samples
30 features
Important

SMOTE was not applied to the test set.

The test set was kept in its original distribution so that model performance could be evaluated on realistic, unseen data.

Train/Test Split

The dataset was divided into:

Training data → 80%
Testing data  → 20%

Training:

X_train: (227845, 30)
y_train: (227845,)

Testing:

X_test: (56962, 30)
y_test: (56962,)
Feature Scaling

A StandardScaler was used to scale the features.

The scaler was fitted on the training data and then used to transform both training and test data.

The trained scaler was saved as:

models/fraud_scaler.pkl
Model

The main machine learning model used in this project is:

XGBoost Classifier

The model was trained on the SMOTE-balanced training data.

The final saved model is:

models/fraud_detection_model.pkl
Model Evaluation

Because this is a highly imbalanced fraud detection problem, accuracy alone is not sufficient.

The following metrics were used:

Precision
Recall
F1-score
ROC-AUC
PR-AUC
Confusion Matrix
Classification Report

Final model performance on the untouched test set:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.60      0.86      0.71        98

    accuracy                           1.00     56962

   macro avg       0.80      0.93      0.85     56962
weighted avg       1.00      1.00      1.00     56962
Fraud Class
Metric	Score
Precision	0.60
Recall	0.86
F1-score	0.71

The model detected approximately 86% of fraud transactions in the test set.

Confusion Matrix

The final confusion matrix is:

[[56808    56]
 [   14    84]]

Interpretation:

True Negatives  = 56808
False Positives = 56
False Negatives = 14
True Positives  = 84

The model correctly detected:

84 out of 98 fraud transactions

The confusion matrix visualization is available in:

confusion_matrix.png
ROC-AUC
ROC-AUC: 0.9796447736382116

Approximately:

ROC-AUC = 0.9796

This indicates that the model has strong ability to distinguish fraudulent transactions from legitimate transactions across different classification thresholds.

PR-AUC
PR-AUC: 0.8649736391160664

Approximately:

PR-AUC = 0.8650

PR-AUC is especially useful for this project because the dataset is highly imbalanced.

Model Tuning

Randomized hyperparameter search was also performed on XGBoost.

The best parameters found were:

subsample        = 1.0
n_estimators     = 200
max_depth        = 5
learning_rate    = 0.1
colsample_bytree = 0.8

However, the tuned model did not improve the PR-AUC compared with the baseline model.

Baseline XGBoost
ROC-AUC: 0.9796447736382116
PR-AUC:  0.8649736391160664
Tuned XGBoost
ROC-AUC: 0.9805622868168089
PR-AUC:  0.8491039510917162

Since PR-AUC decreased after tuning, the baseline XGBoost model was selected as the final model.

Example Fraud Prediction

The saved model was tested using a known fraudulent transaction.

Result:

Prediction: Fraud
Fraud Probability: 0.99995065

The model correctly classified the transaction as fraudulent with a very high predicted probability.

Example Legitimate Prediction

A legitimate transaction was also tested.

Result:

Prediction: Legitimate
Fraud Probability: 0.0001257293

The model correctly classified the transaction as legitimate.

Project Structure
fraud-detection-ml/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── fraud_scaler.pkl
│
├── notebooks/
│   └── frauddetection.ipynb
│
├── src/
│   ├── train_model.py
│   ├── prediction.py
│   ├── evaluate_model.py
│   ├── plot_confusion_matrix.py
│   └── plot_curves.py
│
├── confusion_matrix.png
├── roc_curve.png
├── pr_curve.png
├── requirements.txt
├── results.txt
├── README.md
└── .gitignore
Files Description
train_model.py

Performs the complete training pipeline:

Loads dataset
Separates features and target
Splits data
Scales features
Applies SMOTE
Trains XGBoost
Evaluates the model
Saves model and scaler

Run:

python src/train_model.py
prediction.py

Loads the saved model and scaler and predicts whether a transaction is fraudulent.

Run:

python src/prediction.py

Example:

Prediction: Fraud
Fraud Probability: 0.99995065
evaluate_model.py

Evaluates the already-trained model without retraining.

It produces:

Classification report
Confusion matrix
ROC-AUC
PR-AUC

Run:

python src/evaluate_model.py

This is useful when you want to evaluate the model without running the expensive training process again.

plot_confusion_matrix.py

Generates the confusion matrix visualization.

Run:

python src/plot_confusion_matrix.py

Output:

confusion_matrix.png
plot_curves.py

Generates:

ROC curve
Precision-Recall curve

Run:

python src/plot_curves.py

Output:

roc_curve.png
pr_curve.png
Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:

cd fraud-detection-ml

Install dependencies:

pip install -r requirements.txt
Dataset Setup

The dataset is not included in this repository because of its large file size.

Place the dataset at:

data/creditcard.csv

The expected structure is:

fraud-detection-ml/
└── data/
    └── creditcard.csv
Run the Project
1. Train the model
python src/train_model.py

This performs the complete training pipeline.

2. Evaluate the saved model

After training:

python src/evaluate_model.py

This does not retrain the model.

3. Test a transaction
python src/prediction.py
4. Generate confusion matrix
python src/plot_confusion_matrix.py
5. Generate ROC and PR curves
python src/plot_curves.py
Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Imbalanced-learn
Matplotlib
Joblib
Jupyter Notebook
Key Concepts Demonstrated

This project demonstrates practical understanding of:

Binary classification
Exploratory data analysis
Data preprocessing
Train/test splitting
Feature scaling
Class imbalance
SMOTE
XGBoost
Hyperparameter tuning
Precision
Recall
F1-score
ROC-AUC
PR-AUC
Confusion matrix
Model persistence
Loading trained ML models
Probability-based fraud prediction
Important Note

The model's accuracy is not the main metric for this problem because fraud represents only a small percentage of all transactions.

For fraud detection, precision, recall, F1-score, and PR-AUC provide more meaningful information about model performance.

Future Improvements

Possible future improvements include:

Threshold optimization for fraud detection
Cross-validation with appropriate imbalance handling
Feature engineering
Model comparison with LightGBM and CatBoost
Real-time prediction API using Flask
Web interface for entering transaction details
Model monitoring
Deployment to a cloud platform
Automated retraining pipeline