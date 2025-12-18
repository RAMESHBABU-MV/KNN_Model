import streamlit as st
import joblib
import numpy as np

# 1. Load the model you already trained
# This assumes the file 'knn_model.pkl' is in the same folder as this script
model = joblib.load('knn_model.pkl')

# 2. Set up the UI
st.set_page_config(page_title="Purchase Predictor", page_icon="💰")
st.title("Targeted Ad Predictor")
st.write("Predict whether a customer will purchase a product based on their profile.")

# 3. Create Input Fields
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
with col2:
    salary = st.number_input("Estimated Annual Salary ($)", min_value=1000, max_value=200000, value=50000, step=1000)

# 4. Prediction Logic
if st.button("Predict Outcome"):
    # The model expects a 2D array: [[Age, Salary]]
    features = np.array([[age, salary]])
    
    prediction = model.predict(features)
    
    st.divider()
    if prediction[0] == 1:
        st.success("### Result: Likely to Purchase! ✅")
    else:
        st.error("### Result: Unlikely to Purchase. ❌")