import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor

# --- Load Model and Encoding Maps ---
with open("notebooks/catboost_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("notebooks/manufacturer_means.pkl", "rb") as f:
    manufacturer_means = pickle.load(f)
with open("notebooks/model_means.pkl", "rb") as f:
    model_means = pickle.load(f)
with open("notebooks/region_means.pkl", "rb") as f:
    region_means = pickle.load(f)

# --- Title ---
st.title(" Used Car Price Predictor")
st.caption("Enter your vehicle details to estimate its resale price.")

# --- Layout (2 per row using columns) ---
col1, col2 = st.columns(2)
with col1:
    manufacturer = st.selectbox("Manufacturer", sorted(manufacturer_means.index))
with col2:
    model_name = st.text_input("Model", "corolla")

col3, col4 = st.columns(2)
with col3:
    region = st.text_input("Region", "los angeles")
with col4:
    condition = st.selectbox("Condition", ['excellent', 'good', 'like new', 'fair', 'salvage', 'new', 'missing'])

col5, col6 = st.columns(2)
with col5:
    cylinders = st.selectbox("Cylinders", ['4 cylinders', '6 cylinders', '8 cylinders', 'other', 'missing'])
with col6:
    fuel = st.selectbox("Fuel", ['gas', 'diesel', 'electric', 'hybrid', 'missing'])

col7, col8 = st.columns(2)
with col7:
    title_status = st.selectbox("Title Status", ['clean', 'salvage', 'rebuilt', 'lien', 'missing'])
with col8:
    transmission = st.selectbox("Transmission", ['automatic', 'manual', 'other', 'missing'])

col9, col10 = st.columns(2)
with col9:
    drive = st.selectbox("Drive", ['fwd', 'rwd', '4wd', 'missing'])
with col10:
    car_type = st.selectbox("Type", ['sedan', 'truck', 'SUV', 'wagon', 'pickup', 'other', 'missing'])

col11, col12 = st.columns(2)
with col11:
    paint_color = st.selectbox("Paint Color", ['black', 'white', 'silver', 'red', 'blue', 'grey', 'other', 'missing'])
with col12:
    year = st.slider("Year", 1995, 2025, 2015)

# --- Single full width for numeric ---
odometer = st.number_input("Odometer (in miles)", min_value=0, value=60000)

# --- Predict Button ---
if st.button("Estimate Price"):

    # Clean user input
    region_clean = region.strip().lower()
    model_clean = model_name.strip().lower()

    # Lookup encodings with fallback
    manufacturer_encoded = manufacturer_means.get(manufacturer, manufacturer_means.mean())

    if region_clean in region_means:
        region_encoded = region_means[region_clean]
    else:
        st.warning("️ Region not recognized — using average.")
        region_encoded = region_means.mean()

    if model_clean in model_means:
        model_encoded = model_means[model_clean]
    else:
        st.warning("⚠️ Model not recognized — using average.")
        model_encoded = model_means.mean()

    # Feature engineering
    car_age = 2025 - year
    is_old = int(car_age > 10)
    high_mileage = int(odometer > 150000)
    odometer_log = np.log1p(odometer)
    ppm_log = np.log1p(manufacturer_encoded / (odometer + 1))
    odometer_per_year = odometer / (car_age + 1)
    car_age_squared = car_age ** 2
    age_odometer_interaction = car_age * odometer_log
    log_odometer_times_age = odometer_log * car_age
    odometer_bin = pd.cut([odometer], bins=[0, 30000, 60000, 100000, 150000, 200000, 300000], labels=False)[0]
    if np.isnan(odometer_bin): odometer_bin = 0

    input_df = pd.DataFrame([{
        'model': model_clean,
        'condition': condition,
        'cylinders': cylinders,
        'fuel': fuel,
        'title_status': title_status,
        'transmission': transmission,
        'drive': drive,
        'type': car_type,
        'paint_color': paint_color,
        'car_age': car_age,
        'car_age_squared': car_age_squared,
        'is_old': is_old,
        'high_mileage': high_mileage,
        'manufacturer_encoded': manufacturer_encoded,
        'region_encoded': region_encoded,
        'model_encoded': model_encoded,
        'odometer_log': odometer_log,
        'ppm_log': ppm_log,
        'odometer_per_year': odometer_per_year,
        'age_odometer_interaction': age_odometer_interaction,
        'log_odometer_times_age': log_odometer_times_age,
        'odometer_bin': int(odometer_bin)
    }])

    # Predict
    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)
    st.success(f"Estimated Price: **${predicted_price:,.2f}**")
