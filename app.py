import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Crop Recommendation System", layout="wide")
st.title("🌾 Crop Recommendation System")
st.divider()

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Project Info")
st.sidebar.success("Model Loaded Successfully")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["🌱 Predict Crop", "🖼 Crop Gallery", "📘 Crop Details"])

# ====================================================
# 🌱 PREDICTION TAB
# ====================================================
with tab1:
    left, right = st.columns(2)

    with left:
        temperature = st.slider("🌡 Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 70.0)
        rainfall = st.slider("🌧 Rainfall (mm)", 0.0, 300.0, 150.0)

        N = st.slider("Nitrogen (N)", 0, 140, 90)
        P = st.slider("Phosphorus (P)", 0, 140, 42)
        K = st.slider("Potassium (K)", 0, 140, 43)
        ph = st.slider("pH Level", 0.0, 14.0, 6.5)

        predict_btn = st.button("🚜 Predict Best Crop")

    with right:
        if predict_btn:
            # 1. Define the exact feature names used in your CSV
            feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            
            # 2. Create a DataFrame instead of a raw numpy array
            input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_cols)
            
            # 3. Transform using the DataFrame to silence the warning and ensure accuracy
            scaled_features = scaler.transform(input_df)

            prediction = model.predict(scaled_features)[0]
            st.success(f"🌱 Recommended Crop: {prediction}")

            # ---- Top 3 Crops ----
            try:
                probs = model.predict_proba(scaled_features)[0]
                classes = model.classes_
                top3 = np.argsort(probs)[-3:][::-1]

                st.subheader("🏆 Top 3 Crop Suggestions")
                for i in top3:
                    st.write(f"✔ {classes[i]} — {round(probs[i]*100, 2)}%")
            except:
                st.warning("Probability not supported")

            # ---- Image ----
            img = os.path.join("images", f"{prediction.lower()}.png")
            if os.path.exists(img):
                st.image(img, width=250)
            else:
                st.warning("Crop image not available")
        else:
            st.info("👈 Enter values and click Predict")

# ====================================================
# 🖼 GALLERY TAB
# ====================================================
with tab2:
    csv_file = "Crop_recommendation.csv"

    if not os.path.exists(csv_file):
        st.error("CSV file not found")
        st.stop()

    df = pd.read_csv(csv_file)

    crop_col = None
    for c in ["label", "crop", "Crop", "target", "TARGET"]:
        if c in df.columns:
            crop_col = c
            break

    if crop_col is None:
        st.error("No crop column found")
        st.stop()

    crops = sorted(df[crop_col].unique())
    cols = st.columns(4)

    for i, crop in enumerate(crops):
        img = os.path.join("images", f"{str(crop).lower()}.png")
        with cols[i % 4]:
            if os.path.exists(img):
                st.image(img, caption=str(crop).capitalize(), width="stretch")
            else:
                st.write(str(crop).capitalize())
                st.warning("No image")

# ====================================================
# 📘 CROP DETAILS TAB (AUTO FROM CSV)
# ====================================================
with tab3:
    st.header("📘 Crop Information")

    csv_file = "Crop_recommendation.csv"

    if not os.path.exists(csv_file):
        st.error("CSV file not found")
        st.stop()

    df = pd.read_csv(csv_file)

    crop_col = None
    for c in ["label", "crop", "Crop", "target", "TARGET"]:
        if c in df.columns:
            crop_col = c
            break

    if crop_col is None:
        st.error("No crop column found")
        st.stop()

    crops = sorted(df[crop_col].unique())

    selected_crop = st.selectbox("🌾 Select Crop", crops)

    st.subheader(f"Details for {selected_crop.capitalize()}")

    crop_data = df[df[crop_col] == selected_crop]

    st.write(f"🌡 Avg Temperature: {round(crop_data['temperature'].mean(), 2)} °C")
    st.write(f"💧 Avg Humidity: {round(crop_data['humidity'].mean(), 2)} %")
    st.write(f"🌧 Avg Rainfall: {round(crop_data['rainfall'].mean(), 2)} mm")
    st.write(f"🧪 Avg pH: {round(crop_data['ph'].mean(), 2)}")

    st.write(f"🌱 Nitrogen (N): {round(crop_data['N'].mean(), 2)}")
    st.write(f"🌱 Phosphorus (P): {round(crop_data['P'].mean(), 2)}")
    st.write(f"🌱 Potassium (K): {round(crop_data['K'].mean(), 2)}")

    img = os.path.join("images", f"{selected_crop.lower()}.png")
    if os.path.exists(img):
        st.image(img, width=300)
    else:
        st.warning("Image not available")