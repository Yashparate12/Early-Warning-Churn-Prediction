import pandas as pd
import shap
from src.config import FEATURES

# -----------------------------------
# BUILD FEATURE VECTOR (CRITICAL)
# -----------------------------------
def build_feature_vector(user_inputs: dict) -> pd.DataFrame:
    """
    Builds a FULL feature DataFrame in the exact order
    the model was trained on.
    """

    defaults = {
        'gender': 0,
        'SeniorCitizen': 0,
        'Partner': 0,
        'Dependents': 0,
        'tenure': 12,
        'PhoneService': 1,
        'MultipleLines': 0,
        'InternetService': 1,
        'OnlineSecurity': 0,
        'OnlineBackup': 1,
        'DeviceProtection': 1,
        'TechSupport': 0,
        'StreamingTV': 0,
        'StreamingMovies': 0,
        'Contract': 0,
        'PaperlessBilling': 1,
        'PaymentMethod': 2,
        'MonthlyCharges': 70,
        'TotalCharges': 800
    }

    # Override defaults with user input
    defaults.update(user_inputs)

    return pd.DataFrame(
        [[defaults[f] for f in FEATURES]],
        columns=FEATURES
    )

# -----------------------------------
# PREDICTION
# -----------------------------------
def predict_churn(model, threshold, user_inputs):
    X = build_feature_vector(user_inputs)
    prob = model.predict_proba(X)[0][1]
    churn = int(prob >= threshold)
    return prob, churn

# -----------------------------------
# SHAP EXPLANATION
# -----------------------------------
def get_shap_values(model, input_df):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    return shap_values
