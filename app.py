import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
st.title("House Price Predictor")

# Show how many features the model needs
st.write(f"🔎 Model expects {model.n_features_in_} features")

# Collect exactly TWO inputs
size_sqft = st.number_input("Size (square feet)", min_value=0)
bedrooms  = st.number_input("Number of bedrooms", min_value=0)

if st.button("Predict"):
    features = np.array([[size_sqft, bedrooms]])  # shape (1, 2)
    pred = model.predict(features)[0]
    price = round(float(pred) * 100, 2)
    st.success(f"🏠 Estimated Price: ₹{price}")
