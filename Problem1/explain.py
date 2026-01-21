import joblib
import shap
import pandas as pd

# Load model and data
model = joblib.load("readmission_model.pkl")
X_test = joblib.load("X_test.pkl")
feature_names = joblib.load("feature_names.pkl")

# Convert to DataFrame
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer(X_test_df)

# ---- SELECT ONE PATIENT ----
patient_index = 0  # you can change this

# ---- WATERFALL PLOT (CLASS = 1 → Readmitted) ----
shap.plots.waterfall(
    shap_values[patient_index, :, 1]
)
