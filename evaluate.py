import joblib
from sklearn.metrics import roc_auc_score, classification_report

# Load model and test data
model = joblib.load("readmission_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Evaluation
print("AUROC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
