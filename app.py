import streamlit as st
import numpy as np
import joblib
import json
from PIL import Image
import tensorflow as tf

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kisaan AI 🌾",
    page_icon="🌾",
    layout="centered"
)

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    crop_model      = joblib.load("crop_model.pkl")
    crop_encoder    = joblib.load("crop_encoder.pkl")
    fert_model      = joblib.load("fertilizer_model.pkl")
    soil_enc        = joblib.load("soil_encoder.pkl")
    crop_type_enc   = joblib.load("crop_type_encoder.pkl")
    fert_enc        = joblib.load("fertilizer_encoder.pkl")

    with open("class_names.json") as f:
        class_names = json.load(f)

   interpreter = tf.lite.Interpreter(model_path="disease_model_FINAL.tflite")
    interpreter.allocate_tensors()

    return (crop_model, crop_encoder,
            fert_model, soil_enc, crop_type_enc, fert_enc,
            class_names, interpreter)

(crop_model, crop_encoder,
 fert_model, soil_enc, crop_type_enc, fert_enc,
 class_names, interpreter) = load_models()

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 Kisaan AI")
st.markdown("### AI-Powered Farming Assistant for Indian Farmers")
st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌱 Crop Recommendation",
    "🧪 Fertilizer Recommendation",
    "🍃 Plant Disease Detection"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CROP RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🌱 Crop Recommendation")
    st.markdown("Enter your soil and climate details to find the best crop to grow.")

    col1, col2 = st.columns(2)

    with col1:
        N           = st.number_input("Nitrogen (N)",           0, 200, 50)
        P           = st.number_input("Phosphorus (P)",         0, 200, 50)
        K           = st.number_input("Potassium (K)",          0, 200, 50)
        temperature = st.number_input("Temperature (°C)",       0.0, 60.0, 25.0)

    with col2:
        humidity    = st.number_input("Humidity (%)",           0.0, 100.0, 60.0)
        ph          = st.number_input("Soil pH",                0.0, 14.0,  6.5)
        rainfall    = st.number_input("Rainfall (mm)",          0.0, 500.0, 100.0)

    if st.button("🌾 Recommend Crop", use_container_width=True):
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(features)
        crop_name = crop_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ Recommended Crop: **{crop_name.upper()}**")
        st.balloons()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FERTILIZER RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🧪 Fertilizer Recommendation")
    st.markdown("Enter your soil and crop details to get the best fertilizer.")

    col1, col2 = st.columns(2)

    with col1:
        temp        = st.number_input("Temperature (°C)",   0.0,  60.0, 25.0, key="f_temp")
        moisture    = st.number_input("Moisture (%)",        0.0, 100.0, 50.0)
        rainfall2   = st.number_input("Rainfall (mm)",       0.0, 500.0, 100.0, key="f_rain")
        ph2         = st.number_input("Soil pH",             0.0,  14.0,  6.5,  key="f_ph")

    with col2:
        nitrogen    = st.number_input("Nitrogen (N)",        0, 200, 50,  key="f_n")
        phosphorous = st.number_input("Phosphorous (P)",     0, 200, 50,  key="f_p")
        potassium   = st.number_input("Potassium (K)",       0, 200, 50,  key="f_k")
        carbon      = st.number_input("Carbon (%)",          0.0, 10.0, 1.0)

    soil_types = list(soil_enc.classes_)
    crop_types = list(crop_type_enc.classes_)

    soil_type   = st.selectbox("Soil Type",  soil_types)
    crop_type   = st.selectbox("Crop Type",  crop_types)

    if st.button("🧪 Recommend Fertilizer", use_container_width=True):
        soil_encoded = soil_enc.transform([soil_type])[0]
        crop_encoded = crop_type_enc.transform([crop_type])[0]

        features = np.array([[temp, moisture, rainfall2, ph2,
                               nitrogen, phosphorous, potassium,
                               carbon, soil_encoded, crop_encoded]])

        prediction  = fert_model.predict(features)
        fert_name   = fert_enc.inverse_transform(prediction)[0]

        st.success(f"✅ Recommended Fertilizer: **{fert_name}**")

        # Show remark if available
        remarks = {
            "Urea": "High nitrogen fertilizer. Best for leafy growth.",
            "DAP": "Rich in phosphorus. Great for root development.",
            "MOP": "High potassium. Improves fruit and grain quality.",
            "Compost": "Natural organic fertilizer. Improves soil health.",
        }
        if fert_name in remarks:
            st.info(f"💡 {remarks[fert_name]}")

        st.balloons()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PLANT DISEASE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🍃 Plant Disease Detection")
    st.markdown("Upload a photo of a plant leaf to detect disease.")

    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        if st.button("🔍 Detect Disease", use_container_width=True):
            with st.spinner("Analyzing leaf..."):

                # Preprocess image for TFLite
                img = image.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Run inference
                input_details  = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()

                output = interpreter.get_tensor(output_details[0]['index'])
                pred_index      = str(np.argmax(output))
                confidence      = float(np.max(output)) * 100
                predicted_class = class_names[pred_index]

                # Format display name
                parts       = predicted_class.replace("___", " — ").replace("_", " ")
                plant, condition = parts.split(" — ") if " — " in parts else (parts, "Unknown")

                if "healthy" in predicted_class.lower():
                    st.success(f"✅ Plant: **{plant}**")
                    st.success(f"✅ Status: **Healthy!** (Confidence: {confidence:.1f}%)")
                else:
                    st.warning(f"🌿 Plant: **{plant}**")
                    st.error(f"⚠️ Disease Detected: **{condition}** (Confidence: {confidence:.1f}%)")
                    st.info("💡 Please consult your local agricultural extension officer for treatment advice.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ for Indian Farmers | Kisaan AI 🌾</center>",
    unsafe_allow_html=True
)
