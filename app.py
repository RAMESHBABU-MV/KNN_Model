import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# --- 1. CONFIG & ASSETS ---
st.set_page_config(
    page_title="Purchase Predictor Pro", 
    page_icon="🎯", 
    layout="centered"
)

# Use caching so the model doesn't reload on every interaction
@st.cache_resource
def load_model():
    try:
        return joblib.load('knn_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# --- 2. STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_placeholder=True)

# --- 3. UI LAYOUT ---
st.title("🎯 Targeted Ad Predictor")
st.info("This tool uses Machine Learning (KNN) to predict customer behavior based on demographic data.")

if model is None:
    st.error("⚠️ Error: 'knn_model.pkl' not found. Please ensure the model file is in the script directory.")
    st.stop()

with st.expander("ℹ️ How it works", expanded=False):
    st.write("""
        The model analyzes the **Age** and **Salary** provided and compares them to 
        historical data points to find the most similar customer profiles.
    """)

# Using a container for a cleaner look
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 100, 30)
    with col2:
        salary = st.number_input("Estimated Annual Salary ($)", 1000, 200000, 50000, 500.0)

# --- 4. PREDICTION LOGIC ---
if st.button("Generate Prediction"):
    with st.spinner('Analyzing patterns...'):
        time.sleep(0.5) # Brief pause for UX feel
        
        # Prepare features
        features = np.array([[age, salary]])
        
        # Get prediction and probability (if your model supports it)
        prediction = model.predict(features)[0]
        
        # Try to get probability for a "Confidence Score"
        try:
            prob = model.predict_proba(features)[0]
            confidence = max(prob) * 100
        except:
            confidence = None

    # --- 5. RESULTS DISPLAY ---
    st.subheader("Analysis Result")
    
    if prediction == 1:
        st.success(f"### Likely to Purchase! ✅")
        if confidence:
            st.write(f"Confidence Level: **{confidence:.1f}%**")
    else:
        st.error(f"### Unlikely to Purchase. ❌")
        if confidence:
            st.write(f"Confidence Level: **{confidence:.1f}%**")

    # Visualizing the input relative to a standard range
    st.write("---")
    st.caption("Input Data Summary")
    chart_data = pd.DataFrame([[age, salary / 2000]], columns=['Age', 'Salary (Scaled)'])
    st.bar_chart(chart_data.T)