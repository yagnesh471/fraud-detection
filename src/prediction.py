import pandas as pd
import joblib

# load the trained model
model = joblib.load("models/fraud_detection_model.pkl")
scaler = joblib.load("models/fraud_scaler.pkl")


# function to predict whether a transaction is fraudulent
def predict_fraud(transaction):

    # convert transaction into a DataFrame
    transaction_df = pd.DataFrame(
        [transaction],
        columns=[
            "Time",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
            "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
            "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]
    )

    # scale the transaction
    transaction_scaled = scaler.transform(transaction_df)

    # get fraud probability
    fraud_probability = model.predict_proba(
        transaction_scaled
    )[0][1]

    # make prediction
    prediction = model.predict(transaction_scaled)[0]

    # convert prediction into readable result
    if prediction == 1:
        result = "Fraud"
    else:
        result = "Legitimate"

    # return result and probability
    return result, fraud_probability





# test the prediction function
if __name__ == "__main__":

    # create a sample transaction using the first row from the dataset
    data = pd.read_csv("data/creditcard.csv")

# select the first known fraud transaction
    fraud_row = data[data["Class"] == 1].iloc[0]

# remove the target column
    sample_transaction = fraud_row.drop("Class").tolist()
    # make prediction
    result, probability = predict_fraud(sample_transaction)

    # display the result
    print("Prediction:", result)
    print("Fraud Probability:", probability)