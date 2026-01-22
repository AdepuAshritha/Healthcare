import joblib
import shap
import pandas as pd

model = joblib.load("readmission_model.pkl")
X_test = joblib.load("X_test.pkl")
feature_names = joblib.load("feature_names.pkl")

X_test_df = pd.DataFrame(X_test, columns=feature_names)

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_test_df)

patient_index = 0 

shap.plots.waterfall(
    shap_values[patient_index, :, 1]
)
