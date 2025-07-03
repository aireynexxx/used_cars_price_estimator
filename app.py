import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor, Pool

# --- Load model and encodings ---
with open("models/catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/manufacturer_means.pkl", "rb") as f:
    manufacturer_means = pickle.load(f)

with open("models/model_means.pkl", "rb") as f:
    model_means = pickle.load(f)

with open("models/region_means.pkl", "rb") as f:
    region_means = pickle.load(f)

# --- Define feature columns (must match training order) ---
final_columns = [
    'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',
    'drive', 'type', 'paint_color',
    'car_age', 'car_age_squared', 'is_old', 'high_mileage',
    'manufacturer_encoded', 'region_encoded', 'model_encoded',
    'odometer_log', 'ppm_log', 'odometer_per_year',
    'age_odometer_interaction', 'log_odometer_times_age', 'odometer_bin'
]

categorical_features = [
    'model', 'condition', 'cylinders', 'fuel', 'title_status',
    'transmission', 'drive', 'type', 'paint_color'
]

# --- UI ---
st.set_page_config(page_title="Used Car Price Estimator", layout="centered")
st.title("Used Car Price Estimator")
st.write("Enter the details of a used car to estimate its resale value.")

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    manufacturer = st.selectbox("Manufacturer", sorted(manufacturer_means.index))
    model_name = st.text_input("Model", "corolla")
with col2:
    region = st.text_input("Region", "los angeles")
    condition = st.selectbox("Condition", ['excellent', 'good', 'like new', 'fair', 'salvage', 'new', 'missing'])

col3, col4 = st.columns(2)
with col3:
    cylinders = st.selectbox("Cylinders", ['4 cylinders', '6 cylinders', '8 cylinders', 'other', 'missing'])
    fuel = st.selectbox("Fuel Type", ['gas', 'diesel', 'electric', 'hybrid', 'missing'])
with col4:
    title_status = st.selectbox("Title Status", ['clean', 'salvage', 'rebuilt', 'lien', 'missing'])
    transmission = st.selectbox("Transmission", ['automatic', 'manual', 'other', 'missing'])

col5, col6 = st.columns(2)
with col5:
    drive = st.selectbox("Drive", ['fwd', 'rwd', '4wd', 'missing'])
    car_type = st.selectbox("Type", ['sedan', 'truck', 'SUV', 'wagon', 'pickup', 'other', 'missing'])
with col6:
    paint_color = st.selectbox("Paint Color", ['black', 'white', 'silver', 'red', 'blue', 'grey', 'other', 'missing'])
    year = st.slider("Year", 1995, 2025, 2015)

odometer = st.number_input("Odometer (in miles)", min_value=0, value=60000)

# --- Estimate Price ---
if st.button("Estimate Price"):
    region_clean = region.strip().lower()
    model_clean = model_name.strip().lower()

    manufacturer_encoded = manufacturer_means.get(manufacturer, manufacturer_means.mean())
    model_encoded = model_means.get(model_clean, model_means.mean())
    region_encoded = region_means.get(region_clean, region_means.mean())

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

    # Build input row
    input_data = {
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
    }

    input_df = pd.DataFrame([input_data])[final_columns]

    # Clean up categorical features
    for col in categorical_features:
        input_df[col] = input_df[col].fillna("missing").astype(str)

    # Predict
    try:
        pool = Pool(input_df, cat_features=categorical_features)
        log_price = model.predict(pool)[0]
        predicted_price = np.expm1(log_price)
        st.success(f"Estimated Price: **${predicted_price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
