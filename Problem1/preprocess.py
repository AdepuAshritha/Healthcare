import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("data.csv")

# Separate features and target
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Identify columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Transform data
X_processed = preprocessor.fit_transform(X)

# Save
joblib.dump(X_processed, "X_processed.pkl")
joblib.dump(y, "y.pkl")

joblib.dump(preprocessor, "preprocessor.pkl")
# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Save feature names
joblib.dump(feature_names, "feature_names.pkl")


print("Preprocessing completed successfully.")
