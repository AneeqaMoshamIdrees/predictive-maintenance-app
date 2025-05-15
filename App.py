import streamlit as st
import pickle
import pandas as pd
import os

# Load model
with open('saved_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Check if image exists and then load it
image_path = r'machine_image2.jpg'
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True, caption="Predictive Maintenance System")
else:
    st.warning(f"⚠️ Image not found at path: {image_path}")


st.title("🛠️ Predictive Maintenance Dashboard")

with st.sidebar:
    st.header("Input Sensor Values")
    air_temp = st.number_input("Air Temperature (K)", min_value=200.0, max_value=400.0, value=300.0, help="Current air temperature in Kelvin")
    process_temp = st.number_input("Process Temperature (K)", min_value=200.0, max_value=500.0, value=350.0)
    rot_speed = st.number_input("Rotational Speed (rpm)", min_value=0.0, max_value=4000.0, value=1500.0)
    torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", min_value=0.0, max_value=500.0, value=100.0)

    st.header("Engineered Features")
    air_temp_mean_5 = st.number_input("Air Temp Mean (last 5)", value=air_temp)
    process_temp_std_5 = st.number_input("Process Temp Std (last 5)", value=1.0)
    torque_mean_5 = st.number_input("Torque Mean (last 5)", value=torque)
    rpm_std_5 = st.number_input("RPM Std (last 5)", value=10.0)
    tool_wear_mean_5 = st.number_input("Tool Wear Mean (last 5)", value=tool_wear)

    st.header("Failure Flags (0 = No, 1 = Yes)")
    TWF = st.selectbox("Tool Wear Failure", [0, 1], index=0)
    HDF = st.selectbox("Heat Dissipation Failure", [0, 1], index=0)
    PWF = st.selectbox("Power Failure", [0, 1], index=0)
    OSF = st.selectbox("Overstrain Failure", [0, 1], index=0)
    RNF = st.selectbox("Random Failure", [0, 1], index=0)

input_df = pd.DataFrame([{
    'Air_temperature_K': air_temp,
    'Process_temperature_K': process_temp,
    'Rotational_speed_rpm': rot_speed,
    'Torque_Nm': torque,
    'Tool_wear_min': tool_wear,
    'air_temp_mean_5': air_temp_mean_5,
    'process_temp_std_5': process_temp_std_5,
    'torque_mean_5': torque_mean_5,
    'rpm_std_5': rpm_std_5,
    'tool_wear_mean_5': tool_wear_mean_5,
    'TWF': TWF,
    'HDF': HDF,
    'PWF': PWF,
    'OSF': OSF,
    'RNF': RNF
}])

# Reorder columns to exact order expected by the model to avoid feature_names mismatch
expected_features = model.get_booster().feature_names
input_df = input_df[expected_features]

if st.button("Predict Failure"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]  # probability of failure class
    if prediction[0] == 0:
        st.success(f"✅ All good (no failure). Confidence: {100*(1-proba):.2f}%")
    else:
        st.error(f"🚨 Alert: Failure likely soon! Confidence: {100*proba:.2f}%")

    st.write("### Input Summary")
    st.dataframe(input_df)
