import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="💳 Credit Risk Predictor", layout="wide")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("credit_risk_model.pkl")

# =========================
# CUSTOM CSS (UI IMPROVEMENT)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("💳 Credit Risk Prediction Dashboard")
st.markdown("### 🔍 Predict loan default risk with AI")

# =========================
# LAYOUT (2 COLUMNS)
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Details")

    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Income (₹)", min_value=1000, value=50000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=10000)

    education = st.selectbox("Education Level",
                             ["High School", "Bachelors", "Masters", "PhD"])

    housing = st.selectbox("Housing Status",
                           ["Rent", "Own", "Mortgage"])

with col2:
    st.subheader("📊 Financial Profile")

    credit_score = st.slider("Credit Score", 300, 850, 650)
    employment_years = st.slider("Employment Years", 0, 40, 5)

# =========================
# FEATURE ENGINEERING
# =========================
debt_to_income = loan_amount / income if income > 0 else 0
income_per_year = income / (employment_years + 1)

# =========================
# SHOW INPUT SUMMARY
# =========================
st.markdown("---")
st.subheader("📋 Input Summary")

summary = pd.DataFrame({
    "Feature": ["Age", "Income", "Loan", "Credit Score", "Employment"],
    "Value": [age, income, loan_amount, credit_score, employment_years]
})
st.table(summary)

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Predict Risk"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Loan_Amount": [loan_amount],
        "Credit_Score": [credit_score],
        "Employment_Years": [employment_years],
        "Education_Level": [education],
        "Housing_Status": [housing],
        "Debt_to_Income": [debt_to_income],
        "Income_per_Year": [income_per_year]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # =========================
    # RESULT SECTION
    # =========================
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    prob_default = probability[1]

    # Progress bar (Risk meter)
    st.progress(float(prob_default))

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({prob_default:.2%})")
    else:
        st.success(f"✅ Low Risk ({(1 - prob_default):.2%})")

    # =========================
    # METRICS DISPLAY
    # =========================
    col3, col4 = st.columns(2)

    with col3:
        st.metric("No Default Probability", f"{probability[0]:.2%}")

    with col4:
        st.metric("Default Probability", f"{probability[1]:.2%}")

    # =========================
    # INSIGHTS
    # =========================
    st.markdown("### 💡 Insights")

    if debt_to_income > 0.4:
        st.warning("High Debt-to-Income ratio — risky profile")

    if credit_score < 600:
        st.warning("Low Credit Score — higher chance of default")

    if employment_years < 2:
        st.warning("Low employment stability")

    if income > 80000:
        st.info("Strong income — reduces risk")