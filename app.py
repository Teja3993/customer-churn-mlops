import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from textblob import TextBlob

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Telco Churn AI", page_icon="📡", layout="wide")

st.title("📡 Telco Customer Churn Prediction Engine")
st.markdown("""
This production dashboard utilizes a **Stacking Classifier** (CatBoost, LightGBM, XGBoost + Logistic Regression) 
to predict customer churn probability in real-time. 
""")

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_ml_artifacts():
    # Loading the serialized brains of the project
    model = joblib.load('models/churn_stacking_model.joblib')
    scaler = joblib.load('models/churn_scaler.joblib')
    expected_columns = joblib.load('models/expected_columns.joblib')
    num_cols = joblib.load('models/num_cols.joblib')
    return model, scaler, expected_columns, num_cols

model, scaler, expected_columns, num_cols = load_ml_artifacts()

# --- AUTOMATED COMPLAINT GUESSER ---
def guess_complaint(text):
    """
    Simulates the SLM extraction logic using lightweight keyword triggers.
    This ensures the app stays responsive on i3 hardware.
    """
    text = text.lower()
    if any(word in text for word in ["price", "cost", "bill", "charge", "expensive", "money", "pay"]):
        return "Pricing"
    if any(word in text for word in ["slow", "network", "signal", "internet", "drop", "speed", "wifi", "connection"]):
        return "Network"
    if any(word in text for word in ["support", "person", "help", "service", "rude", "call", "agent", "wait"]):
        return "Support"
    if any(word in text for word in ["competitor", "other", "switch", "provider", "verizon", "at&t", "t-mobile"]):
        return "Competitor"
    return "None"

# --- SIDEBAR INPUTS ---
st.sidebar.header("Customer Profile")

# 1. Demographics
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

# 2. Account & Billing
st.sidebar.subheader("Account & Billing")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", value=70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", value=float(tenure * monthly_charges))
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# 3. Services
st.sidebar.subheader("Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# 4. NLP & Feedback
st.sidebar.subheader("Customer Feedback (NLP)")
feedback = st.sidebar.text_area("Recent Customer Feedback", "The internet is too slow and costs too much.")

# Auto-update the dropdown based on text input
suggested_complaint = guess_complaint(feedback)
complaint_options = ["Pricing", "Network", "Support", "Competitor", "None"]

primary_complaint = st.sidebar.selectbox(
    "Primary Complaint (AI Suggested)", 
    complaint_options,
    index=complaint_options.index(suggested_complaint)
)

# --- PREPROCESSING & PREDICTION ---
if st.sidebar.button("Predict Churn Risk", type="primary"):
    
    # 1. Calculate Sentiment Score dynamically via TextBlob
    sentiment_score = TextBlob(feedback).sentiment.polarity
    
    # 2. Construct raw dataframe
    input_dict = {
        'gender': gender, 'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges, 
        'Primary_Complaint': primary_complaint, 'Sentiment_Score': sentiment_score
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # 3. One-Hot Encoding
    cat_features = [col for col in input_df.columns if col not in num_cols and col != 'Sentiment_Score']
    input_encoded = pd.get_dummies(input_df, columns=cat_features)
    
    # 4. Clean column names (LightGBM JSON Fix)
    input_encoded = input_encoded.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    
    # 5. Align with Model's Expected Columns
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    input_encoded = input_encoded[expected_columns]
    
    # 6. Scale Numerical Features
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
    
    # 7. Generate Prediction
    churn_prob = model.predict_proba(input_encoded)[0][1]
    
    # --- DISPLAY RESULTS ---
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Churn Probability", value=f"{churn_prob * 100:.2f}%")
        st.progress(float(churn_prob))
        
    with col2:
        if churn_prob > 0.60:
            st.error("🚨 HIGH RISK: Immediate retention intervention required.")
        elif churn_prob > 0.40:
            st.warning("⚠️ MEDIUM RISK: Monitor account activity.")
        else:
            st.success("✅ LOW RISK: Customer is currently stable.")
            
    st.write("---")
    st.info(f"**AI Analysis:** Detected **{primary_complaint}** issue with a sentiment score of **{sentiment_score:.2f}**.")