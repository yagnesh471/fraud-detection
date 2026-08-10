import pandas as pd

# load the full dataset
df = pd.read_csv("data/creditcard.csv")

# separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# display the shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

# display feature names
print("\nFeatures:")
print(X.columns)




from sklearn.model_selection import train_test_split

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# display the shapes
print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)




from sklearn.preprocessing import StandardScaler

# create the scaler
scaler = StandardScaler()

# learn scaling parameters only from training data
X_train_scaled = scaler.fit_transform(X_train)

# use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)

# display the shapes
print("\nScaled data:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)




from imblearn.over_sampling import SMOTE

# create the SMOTE object
smote = SMOTE(random_state=42)

# apply SMOTE only to the training data
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

# display class distribution after SMOTE
print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# display the new training shape
print("\nTraining data after SMOTE:")
print("X_train_smote:", X_train_smote.shape)
print("y_train_smote:", y_train_smote.shape)




from xgboost import XGBClassifier

# create the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

# train the model using SMOTE-balanced data
xgb_model.fit(
    X_train_smote,
    y_train_smote
)

print("\nXGBoost training completed!")



from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# make predictions on the test data
y_pred = xgb_model.predict(X_test_scaled)

# get fraud probabilities
y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

# display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

# calculate PR-AUC
pr_auc = average_precision_score(y_test, y_prob)

print("XGBoost ROC-AUC:", roc_auc)
print("XGBoost PR-AUC:", pr_auc)



from sklearn.model_selection import RandomizedSearchCV

# create the XGBoost model for tuning
xgb_tuning = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# define the hyperparameter search space
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# randomized hyperparameter search
random_search = RandomizedSearchCV(
    estimator=xgb_tuning,
    param_distributions=param_grid,
    n_iter=10,
    scoring="average_precision",
    cv=3,
    random_state=42,
    n_jobs=-1
)

# train and search for the best parameters
random_search.fit(
    X_train_smote,
    y_train_smote
)

# display the best parameters
print("\nBest Parameters:")
print(random_search.best_params_)

# get the best model
xgb_tuned = random_search.best_estimator_



# make predictions using the tuned model
y_pred_tuned = xgb_tuned.predict(X_test_scaled)

# get fraud probabilities
y_prob_tuned = xgb_tuned.predict_proba(X_test_scaled)[:, 1]

# display classification report
print("\nTuned XGBoost Classification Report:")
print(classification_report(y_test, y_pred_tuned))

# calculate ROC-AUC
tuned_roc_auc = roc_auc_score(y_test, y_prob_tuned)

# calculate PR-AUC
tuned_pr_auc = average_precision_score(y_test, y_prob_tuned)

print("Tuned XGBoost ROC-AUC:", tuned_roc_auc)
print("Tuned XGBoost PR-AUC:", tuned_pr_auc)



import joblib

# save the baseline XGBoost model
joblib.dump(
    xgb_model,
    "models/fraud_detection_model.pkl"
)

# save the scaler
joblib.dump(
    scaler,
    "models/fraud_scaler.pkl"
)

print("\nModel and scaler saved successfully!")



