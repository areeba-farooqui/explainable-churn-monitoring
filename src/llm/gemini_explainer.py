
import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def explain_churn_and_drift(
    customer_prediction,
    drift_report,
    rca_report,
    global_shap_importance=None
):
    """Explain churn predictions and monitoring results using Gemini."""

    if isinstance(rca_report, pd.DataFrame):
        rca_report = rca_report.to_dict(orient="records")

    prompt = f"""
    You are an AI assistant for a customer churn monitoring system.

    CUSTOMER CHURN PREDICTION:
    {json.dumps(customer_prediction, default=str)}

    DRIFT REPORT:
    {json.dumps(drift_report, default=str)}

    ROOT CAUSE ANALYSIS:
    {json.dumps(rca_report, default=str)}

    GLOBAL SHAP IMPORTANCE:
    {json.dumps(global_shap_importance or {}, default=str)}

    IMPORTANT RULES:

    1. Use the customer prediction to explain whether the customer
       is predicted to churn or stay. Report the supplied probability.

    2. Global SHAP importance describes which features matter
       generally to the model. It does not explain an individual
       customer's prediction.

    3. RCA identifies features associated with data drift.
       It does not necessarily explain why an individual customer
       will churn.

    4. Drift means a change in data distribution. It does not
       automatically mean a customer will churn.

    5. Do not invent customer details, probabilities, or causes.

    Explain:
    - The customer's churn prediction and probability.
    - What is known and unknown about the individual prediction.
    - The most important global model features.
    - The detected drift and RCA findings.
    - Practical recommendations for investigation and retention.

    Return valid JSON with exactly these keys:
    summary, churn_explanation, global_model_factors,
    drift_explanation, rca_findings, recommendations, limitations.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)