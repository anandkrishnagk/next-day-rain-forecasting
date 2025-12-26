import streamlit as st
import pickle
import numpy as np
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Next-Day Rainfall Forecasting",
    page_icon="🌧️",
    layout="wide"
)

# ---------------------------------------------------
# GLOBAL CSS (MATCHES DESIGN IMAGE)
# ---------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: White;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
}

button[kind="primary"] {
    width: 100%;
    height: 55px;
    font-size: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------
ohe = pickle.load(open("onehot_encoder.sav", "rb"))
std = pickle.load(open("scaler.sav", "rb"))
model = pickle.load(open("rain_classifier.sav", "rb"))

location_list = ohe.categories_[0].tolist()
wind_dirs = ohe.categories_[1].tolist()

# ---------------------------------------------------
# HERO SECTION (FINAL – FIXED)
# ---------------------------------------------------
hero_left, hero_right = st.columns([1.4, 1])

with hero_left:
    st.markdown(
        '<div style="background: linear-gradient(135deg, #eef4ff, #f9fcff);'
        'padding: 40px; border-radius: 20px;">'
        '<h1>🌧️ Next-Day Rainfall Forecasting</h1>'
        '<p style="font-size:17px;">'
        'ML-based weather forecasting using today’s atmospheric data'
        '</p>'
        '<h2 style="margin-top:25px;">'
        'Will it rain tomorrow in your area?'
        '</h2>'
        '<p style="font-size:16px; line-height:1.6;">'
        'Enter today’s temperature, humidity, wind, and pressure values to get a'
        'data-driven prediction for tomorrow’s rainfall.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

with hero_right:
    st.image("hero_background.jpeg", width=720)


# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------
input_col, result_col = st.columns([1.4, 1])

# ---------------------------------------------------
# INPUT WEATHER DATA
# ---------------------------------------------------
with input_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Input Weather Data")

    with st.expander("🌡️ Temperature & Rainfall", expanded=True):
        MinTemp = st.number_input("Minimum Temperature (°C)")
        MaxTemp = st.number_input("Maximum Temperature (°C)")
        Rainfall = st.number_input("Rainfall (mm)")

    with st.expander("💨 Wind Parameters"):
        WindGustSpeed = st.number_input("Wind Gust Speed (km/h)")
        WindSpeed9am = st.number_input("Wind Speed at 9 AM (km/h)")
        WindSpeed3pm = st.number_input("Wind Speed at 3 PM (km/h)")

    with st.expander("💧 Humidity & Pressure"):
        Humidity9am = st.number_input("Humidity at 9 AM (%)")
        Humidity3pm = st.number_input("Humidity at 3 PM (%)")
        Pressure9am = st.number_input("Pressure at 9 AM (hPa)")
        Pressure3pm = st.number_input("Pressure at 3 PM (hPa)")
        Temp9am = st.number_input("Temperature at 9 AM (°C)")
        Temp3pm = st.number_input("Temperature at 3 PM (°C)")

    with st.expander("📍 Location & Rain History"):
        Location_In = st.selectbox("Select Location", location_list)
        WindGustDir_In = st.selectbox("Wind Gust Direction", wind_dirs)
        WindDir9am_In = st.selectbox("Wind Direction at 9 AM", wind_dirs)
        WindDir3pm_In = st.selectbox("Wind Direction at 3 PM", wind_dirs)
        RainToday = st.radio("Did it rain today?", ["No", "Yes"])
        RainToday = 1 if RainToday == "Yes" else 0

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# FORECAST RESULT
# ---------------------------------------------------
with result_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Forecast Result")
    st.info("👉 Enter weather data and click **Predict Rainfall**")

    if st.button("🌧️ Predict Rainfall", type="primary"):
        # Encode categorical inputs
        ohe_input = ohe.transform([[Location_In, WindGustDir_In, WindDir9am_In, WindDir3pm_In]])

        # Combine features
        final_features = np.concatenate([
            [MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am, WindSpeed3pm,
             Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
             Temp9am, Temp3pm, RainToday],
            ohe_input[0]
        ]).reshape(1, -1)

        # Scale numerical features
        final_features[:, :12] = std.transform(final_features[:, :12])

        prediction = model.predict(final_features)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.error("🌧️ **Rain Expected Tomorrow**")
        else:
            st.success("☀️ **No Rain Expected Tomorrow**")

    st.markdown('</div>', unsafe_allow_html=True)
