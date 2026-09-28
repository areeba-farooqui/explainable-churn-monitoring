import sys
import json
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import encode_for_model
from src.llm.gemini_explainer import explain_churn_and_drift

# Results folder
TABLES = ROOT / "results" / "tables"

# Load actual monitoring results
drift_df = pd.read_csv(TABLES / "drift_results.csv")
rca_df = pd.read_csv(TABLES / "rca_results.csv")

with open(TABLES / "shap_importance.json", "r") as f:
    global_shap = json.load(f)

# Select scenario S1, seed 0
scenario = "S1"
seed = 0

# Filter drift results
drift_report = drift_df[
    (drift_df["scenario"] == scenario) &
    (drift_df["seed"] == seed)
].to_dict(orient="records")

# Filter RCA results
rca_report = rca_df[
    (rca_df["scenario"] == scenario) &
    (rca_df["seed"] == seed)
].to_dict(orient="records")

# Demo prediction only — NOT a real customer prediction
# Load a real customer from the current dataset
current_df = pd.read_csv(ROOT / "data" / "raw" / "current.csv")
customer = current_df.iloc[[0]]

# Prepare features for the trained model
X = encode_for_model(customer).drop(columns=["Churn"])

features = pd.read_csv(ROOT / "artifacts" / "feature_names.csv").iloc[:, 0].tolist()
X = X.reindex(columns=features, fill_value=0)

# Load the trained model
model = xgb.XGBClassifier()
model.load_model(str(ROOT / "artifacts" / "xgb_model.json"))

# Generate the actual prediction
prediction = int(model.predict(X)[0])
probability = float(model.predict_proba(X)[0][1])

customer_prediction = {
    "prediction": "Churn" if prediction == 1 else "No churn",
    "probability": round(probability, 4)
}

# Send results to Gemini
result = explain_churn_and_drift(
    customer_prediction=customer_prediction,
    drift_report=drift_report,
    rca_report=rca_report,
    global_shap_importance=global_shap
)

# Print formatted JSON response
print(json.dumps(result, indent=2))