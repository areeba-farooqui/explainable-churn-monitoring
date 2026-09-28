import sys
import json
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Set project paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Import project functions
from src.data.preprocessing import encode_for_model
from monitoring.drift_metrics import detect_drift_all_features
from monitoring.rca import compute_rca
from src.llm.gemini_explainer import explain_churn_and_drift


# Define paths
DATA = ROOT / "data" / "raw"
TABLES = ROOT / "results" / "tables"
ARTIFACTS = ROOT / "artifacts"


# --------------------------------------------------
# 1. Load reference and current data
# --------------------------------------------------

reference_df = pd.read_csv(DATA / "reference.csv")
current_df = pd.read_csv(DATA / "current.csv")


# --------------------------------------------------
# 2. Define feature lists
# --------------------------------------------------

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# --------------------------------------------------
# 3. Run actual drift detection
# --------------------------------------------------

drift_report = detect_drift_all_features(
    reference_df,
    current_df,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES
)


# --------------------------------------------------
# 4. Load global SHAP importance
# --------------------------------------------------

with open(TABLES / "shap_importance.json", "r") as f:
    global_shap = json.load(f)


# --------------------------------------------------
# 5. Run actual Root Cause Analysis
# --------------------------------------------------

rca_report = compute_rca(
    drift_report,
    global_shap
)


# --------------------------------------------------
# 6. Generate a real customer prediction
# --------------------------------------------------

customer = current_df.iloc[[0]]

X = encode_for_model(customer).drop(columns=["Churn"])

features = (
    pd.read_csv(ARTIFACTS / "feature_names.csv")
    .iloc[:, 0]
    .tolist()
)

X = X.reindex(columns=features, fill_value=0)

model = xgb.XGBClassifier()
model.load_model(str(ARTIFACTS / "xgb_model.json"))

prediction = int(model.predict(X)[0])
probability = float(model.predict_proba(X)[0][1])

customer_prediction = {
    "prediction": "Churn" if prediction == 1 else "No churn",
    "probability": round(probability, 4)
}


# --------------------------------------------------
# 7. Send everything to Gemini
# --------------------------------------------------

result = explain_churn_and_drift(
    customer_prediction=customer_prediction,
    drift_report=drift_report,
    rca_report=rca_report,
    global_shap_importance=global_shap
)


# --------------------------------------------------
# 8. Display the explanation
# --------------------------------------------------

print("\nCustomer Prediction:")
print(json.dumps(customer_prediction, indent=2))

print("\nGemini Explanation:")
print(json.dumps(result, indent=2, ensure_ascii=False))