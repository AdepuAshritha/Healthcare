import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


X = joblib.load("X_processed.pkl")
y = joblib.load("y.pkl")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)


model.fit(X_train, y_train)


joblib.dump(model, "readmission_model.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

print("Model training completed.")
