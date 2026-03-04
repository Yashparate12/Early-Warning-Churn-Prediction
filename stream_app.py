import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.predict import predict_churn, get_shap_values, build_feature_vector

# Fix matplotlib backend for Streamlit
matplotlib.use("Agg")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Churn Early Warning System",
    layout="wide"
)

st.title("📉 Customer Churn Early Warning System")
st.caption("Early detection • Explainable AI • Prevent churn before it happens")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/churn_model.pkl")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# Stop if model not loaded
if model is None:
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧑 Customer Profile (Early Stage)")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 6)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    0,
    150,
    85
)

total_charges = st.sidebar.number_input(
    "Total Charges ($)",
    min_value=0.0,
    value=float(tenure * monthly_charges),
    step=1.0
)

# ---------------- EARLY WARNING THRESHOLD ----------------
st.sidebar.subheader("🎯 Early Warning Sensitivity")

early_threshold = st.sidebar.slider(
    "Alert Sensitivity (Lower = Earlier Warning)",
    min_value=0.2,
    max_value=0.6,
    value=0.35,
    step=0.05
)

# ---------------- INPUT MODE ----------------
st.subheader("⚙️ Input Mode")

input_mode = st.radio(
    "Choose input strategy",
    ["Auto (Recommended)", "Manual (What-if Analysis)"],
    horizontal=True
)

disable_manual = input_mode == "Auto (Recommended)"

# ---------------- CUSTOMER ATTRIBUTES ----------------
st.subheader("📋 Customer Attributes")
st.caption("Auto-filled by default; editable in Manual mode for what-if analysis")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"], disabled=disable_manual)
    senior = st.selectbox("Senior Citizen", ["No", "Yes"], disabled=disable_manual)
    partner = st.selectbox("Partner", ["Yes", "No"], disabled=disable_manual)
    dependents = st.selectbox("Dependents", ["No", "Yes"], disabled=disable_manual)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"], disabled=disable_manual)
    multiline = st.selectbox("Multiple Lines", ["Yes", "No"], disabled=disable_manual)
    internet = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"], disabled=disable_manual)
    online_security = st.selectbox("Online Security", ["Yes", "No"], disabled=disable_manual)

with col3:
    tech_support = st.selectbox("Tech Support", ["Yes", "No"], disabled=disable_manual)
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], disabled=disable_manual)
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"], disabled=disable_manual)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], disabled=disable_manual)

# ---------------- ENCODING ----------------
binary_map = {"No": 0, "Yes": 1}
gender_map = {"Female": 1, "Male": 0}
internet_map = {"DSL": 0, "Fiber Optic": 2, "No": 1}

# ---------------- FEATURE DICTIONARY ----------------
user_inputs = {
    "tenure": float(tenure),
    "MonthlyCharges": float(monthly_charges),
    "TotalCharges": float(total_charges),

    "gender": gender_map.get(gender, 0),
    "SeniorCitizen": binary_map.get(senior, 0),
    "Partner": binary_map.get(partner, 0),
    "Dependents": binary_map.get(dependents, 0),

    "PhoneService": binary_map.get(phone, 0),
    "MultipleLines": binary_map.get(multiline, 0),
    "InternetService": internet_map.get(internet, 0),
    "OnlineSecurity": binary_map.get(online_security, 0),

    "OnlineBackup": 1,
    "DeviceProtection": 1,
    "TechSupport": binary_map.get(tech_support, 0),
    "StreamingTV": binary_map.get(streaming_tv, 0),
    "StreamingMovies": binary_map.get(streaming_movies, 0),

    "Contract": 0,
    "PaperlessBilling": binary_map.get(paperless, 0),
    "PaymentMethod": 2
}

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔮 Predict Churn Risk"):

    try:
        prob, _ = predict_churn(model, early_threshold, user_inputs)

        # Risk logic
        if prob >= early_threshold + 0.2:
            risk = "HIGH"
        elif prob >= early_threshold:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # ---------------- RESULTS ----------------
        st.subheader("📊 Prediction Result")

        st.write(f"**Churn Probability:** {prob*100:.2f}%")
        st.write(f"**Risk Category:** {risk}")

        if risk == "HIGH":
            st.error("🚨 Customer shows strong early signs of churn")
        elif risk == "MEDIUM":
            st.warning("⚠️ Customer may churn without intervention")
        else:
            st.success("✅ Customer is currently stable")

        # ---------------- EXPLANATION ----------------
        st.subheader("🧠 Why This Is an EARLY Warning")

        st.write("""
        This system detects disengagement **before cancellation occurs**.
        It favors **recall over accuracy**, ensuring at-risk customers
        are flagged early for proactive retention.
        """)

        # ---------------- KEY DRIVERS ----------------
        st.subheader("🔍 Key Risk Drivers")

        if tenure < 12:
            st.write("• Short tenure → weak customer attachment")

        if monthly_charges > 80:
            st.write("• High monthly charges → value sensitivity risk")

        st.write("• Month-to-month contracts → easy exit behavior")

        # ---------------- SHAP ----------------
        st.subheader("📈 Model Explainability (SHAP)")

        input_df = build_feature_vector(user_inputs)
        shap_values = get_shap_values(model, input_df)

        fig, ax = plt.subplots(figsize=(9,4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close()

        # ---------------- WHAT IF ----------------
        st.subheader("📊 What-If Scenario Comparison")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 🔴 Current Scenario")
            st.metric("Churn Risk", f"{prob*100:.1f}%")

        with colB:
            st.markdown("### 🟢 Improved Scenario")

            improved_inputs = user_inputs.copy()
            improved_inputs["MonthlyCharges"] *= 0.90
            improved_inputs["Contract"] = 1
            improved_inputs["TechSupport"] = 1

            improved_prob, _ = predict_churn(model, early_threshold, improved_inputs)

            st.metric(
                "Churn Risk",
                f"{improved_prob*100:.1f}%",
                delta=f"-{(prob-improved_prob)*100:.1f}%"
            )

        # ---------------- ACTIONS ----------------
        st.subheader("🛠️ Early Retention Actions")

        if risk == "HIGH":
            st.write("""
            - Immediate retention offers  
            - Contract upgrade incentives  
            - Priority customer outreach  
            """)

        elif risk == "MEDIUM":
            st.write("""
            - Proactive engagement campaigns  
            - Value reinforcement  
            """)

        else:
            st.write("""
            - Loyalty rewards  
            - Maintain engagement  
            """)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Early Warning Churn System • Explainable ML • Business-Driven AI")