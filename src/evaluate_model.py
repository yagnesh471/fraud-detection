import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# load the saved model
model = joblib.load("models/fraud_detection_model.pkl")

# load the saved scaler
scaler = joblib.load("models/fraud_scaler.pkl")

# load the original dataset
df = pd.read_csv("data/creditcard.csv")

# separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# scale the test data using the saved scaler
X_test_scaled = scaler.transform(X_test)

# make predictions
y_pred = model.predict(X_test_scaled)

# get fraud probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

# PR-AUC
pr_auc = average_precision_score(y_test, y_prob)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)