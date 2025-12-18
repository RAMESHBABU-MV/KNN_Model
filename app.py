import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Purchase Predictor Pro", 
    page_icon="🎯", 
    layout="centered"
)

# --- 2. MODEL LOADING (With Caching) ---
@st.cache_resource
def load_model():
    try:
        # Ensure 'knn_model.pkl' is in the same folder
        return joblib.load('knn_model.pkl')
    except Exception as e:
        return None

model = load_model()

# --- 3. CUSTOM STYLING ---
# Fixed the 'unsafe_allow_html' parameter here
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        background-color: #007bff; 
        color: white;
        font-weight: bold;
    }
    .stNumberInput, .stSlider {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.title("🎯 Targeted Ad Predictor")
st.write("Determine if a customer is likely to purchase based on age and estimated income.")
st.divider()

# Check if model loaded successfully
if model is None:
    st.error("❌ **Model File Not Found:** Please ensure `knn_model.pkl` is in your repository/folder.")
    st.info("If you are running this locally, check your file path. If on Streamlit Cloud, make sure you committed the .pkl file to GitHub.")
    st.stop()

# --- 5. USER INPUTS ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Customer Age", min_value=18, max_value=100, value=30)
    
    with col2:
        salary = st.number_input("Annual Salary ($)", min_value=1000, max_value=200000, value=50000, step=1000)

# --- 6. PREDICTION LOGIC ---
if st.button("Analyze Customer Profile"):
    # Show a progress bar for visual effect
    with st.spinner('Calculating probabilities...'):
        time.sleep(0.6)
        
        # Format input for the model
        features = np.array([[age, salary]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate confidence if the model supports it
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = np.max(probabilities) * 100
        except:
            confidence = None

    # --- 7. DISPLAY RESULTS ---
    st.subheader("Result:")
    
    if prediction == 1:
        st.success("### ✅ Likely to Purchase")
        st.balloons()
    else:
        st.error("### ❌ Unlikely to Purchase")

    # Display Confidence Score if available
    if confidence:
        st.write(f"**Model Confidence:** {confidence:.2f}%")
        st.progress(confidence / 100)

    # Simple Visual Context
    st.divider()
    st.caption("Input Summary Reference")
    summary_df = pd.DataFrame({"Metric": ["Age", "Salary"], "Value": [age, salary]})
    st.table(summary_df)