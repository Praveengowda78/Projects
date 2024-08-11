import streamlit as st
import joblib
import numpy as np

# Load your trained model and scaler
model = joblib.load('SVM_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Credit Card Fraud Detection')

# Input fields for user to enter transaction details
transaction_amount = st.number_input('Transaction Amount', min_value=0.0, step=0.01)
transaction_time = st.number_input('Transaction Time', min_value=0.0, step=0.01)
# Add more input fields as per your model's features

# Button to make prediction
if st.button('Predict'):
    # Prepare input features for the model
    features = np.array([transaction_amount, transaction_time])  # Update based on your features
    features = features.reshape(1, -1)
    
    # Apply the same scaler used during training
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)[:, 1]
    
    # Display result
    result = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'
    st.write(f'Prediction: {result}')
    st.write(f'Probability of being Fraudulent: {prediction_proba[0]:.2f}')
