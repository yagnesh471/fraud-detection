import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Starting confusion matrix generation...")

# Load model
model = joblib.load("models/fraud_detection_model.pkl")

# Load scaler
scaler = joblib.load("models/fraud_scaler.pkl")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Create plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

disp.plot()

plt.title("Fraud Detection - Confusion Matrix")
plt.tight_layout()

# Save plot
plt.savefig("confusion_matrix.png", dpi=300)

print("\nConfusion matrix saved successfully!")
print("File: confusion_matrix.png")