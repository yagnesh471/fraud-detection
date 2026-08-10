import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

# load saved model
model = joblib.load("models/fraud_detection_model.pkl")

# load saved scaler
scaler = joblib.load("models/fraud_scaler.pkl")

# load dataset
df = pd.read_csv("data/creditcard.csv")

# separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# scale test data
X_test_scaled = scaler.transform(X_test)

# get fraud probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# ROC Curve
# -----------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fraud Detection - ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig("roc_curve.png", dpi=300)

plt.show()


# -----------------------------
# Precision-Recall Curve
# -----------------------------

precision, recall, _ = precision_recall_curve(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)

plt.figure()
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Fraud Detection - Precision-Recall Curve")
plt.legend()
plt.tight_layout()

plt.savefig("pr_curve.png", dpi=300)

plt.show()

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)