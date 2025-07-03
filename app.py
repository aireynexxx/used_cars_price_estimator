import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor

# --- Load trained model ---
with open("models/catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Load mean encodings and convert to dicts if needed ---
with open("models/manufacturer_means.pkl", "rb") as f:
    manufacturer_means = pickle.load(f)
    manufacturer_means = dict(manufacturer_means)

with open("models/model_means.pkl", "rb") as f:
    model_means = pickle.load(f)
    model_means = dict(model_means)

# --- Dropdown options ---
manufacturer_options = sorted(manufacturer_means.keys())

# --- App Title ---
st.set_page_config(page_title="Used Car Price Estimator", layout="centered")
st.title("Used Car Price Estimator")
st.write("Enter the details of a used car to get an estimated resale price.")

# --- Input Layout ---
col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox("Manufacturer", manufacturer_options).lower().strip()
    car_model = st.text_input("Model (e.g., civic, f-150)").lower().strip()
    condition = st.selectbox("Condition", ['excellent', 'good', 'like new', 'fair', 'new', 'salvage'])
    cylinders = st.selectbox("Cylinders", ['4 cylinders', '6 cylinders', '8 cylinders', '3 cylinders', '5 cylinders', '10 cylinders', '12 cylinders', 'other'])

with col2:
    fuel = st.selectbox("Fuel Type", ['gas', 'diesel', 'electric', 'hybrid', 'other'])
    transmission = st.selectbox("Transmission", ['automatic', 'manual', 'other'])
    drive = st.selectbox("Drive", ['fwd', 'rwd', '4wd'])
    paint_color = st.selectbox("Paint Color", ['white', 'black', 'silver', 'blue', 'red', 'grey', 'green', 'custom', 'brown', 'yellow', 'orange', 'purple'])

# --- More Inputs ---
col3, col4 = st.columns(2)

with col3:
    title_status = st.selectbox("Title Status", ['clean', 'rebuilt', 'salvage', 'lien', 'missing', 'parts only'])
    type_ = st.selectbox("Vehicle Type", ['sedan', 'SUV', 'pickup', 'truck', 'coupe', 'wagon', 'van', 'convertible', 'mini-van', 'hatchback', 'offroad', 'bus', 'other'])

with col4:
    odometer = st.number_input("Odometer (miles)", min_value=0, value=60000)
    year = st.slider("Year of Manufacture", min_value=1980, max_value=2025, value=2015)

# --- Estimate Price ---
if st.button("Estimate Price"):
    # --- Feature Engineering ---
    car_age = 2025 - year
    is_old = int(car_age > 10)
    high_mileage = int(odometer > 150000)
    odometer_log = np.log1p(odometer)

    # Mean encodings with fallback
    manufacturer_encoded = manufacturer_means[manufacturer] if manufacturer in manufacturer_means else np.mean(list(manufacturer_means.values()))
    model_encoded = model_means[car_model] if car_model in model_means else np.mean(list(model_means.values()))
    est_price = manufacturer_encoded
    ppm_log = np.log1p(est_price / (odometer + 1))

    # --- Prepare input row ---
    input_df = pd.DataFrame([{
        'manufacturer': manufacturer,
        'model': car_model,
        'condition': condition,
        'cylinders': cylinders,
        'fuel': fuel,
        'title_status': title_status,
        'transmission': transmission,
        'drive': drive,
        'type': type_,
        'paint_color': paint_color,
        'car_age': car_age,
        'is_old': is_old,
        'high_mileage': high_mileage,
        'manufacturer_encoded': manufacturer_encoded,
        'model_encoded': model_encoded,
        'odometer_log': odometer_log,
        'ppm_log': ppm_log
    }])

    # --- Predict ---
    y_pred_log = model.predict(input_df)[0]
    y_pred = np.expm1(y_pred_log)
    st.success(f"Estimated Price: **${y_pred:,.2f}**")
