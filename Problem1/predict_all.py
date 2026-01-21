import joblib

# Load trained model and test data
model = joblib.load("readmission_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")  # actual outcomes (for comparison)

# Get predictions
risk_probabilities = model.predict_proba(X_test)[:, 1]
binary_predictions = model.predict(X_test)

print("Patient-wise Readmission Predictions")
print("------------------------------------")

for i in range(len(X_test)):
    print(
        f"Patient {i+1}: "
        f"Risk Probability = {risk_probabilities[i]:.2f}, "
        f"Predicted = {binary_predictions[i]} "
       
    )
