import streamlit as st
import os, joblib, json, tempfile
from PIL import Image
from src.utils.config import Weights
from src.utils.model_utils import load_cv_model
from src.models.infer import load_sensor_model, combined_inference


st.set_page_config(page_title="Nutmeg Fungal Early Warning", layout="centered")

st.title("🌿 Nutmeg Fungal Disease Early Warning")
st.write("IoT-like Sensors + Image Classification → Risk Score + Recommendations")

with st.sidebar:
    st.header("Models")
    sensor_model_path = st.text_input("Sensor model path", "models/sensor_model.pkl")
    cv_model_path = st.text_input("CV model path", "models/cv_model.pt")
    class_to_idx_path = st.text_input("Class mapping path", "models/class_to_idx.json")
    st.write("Weights for ensemble (0-1)")
    sensor_w = st.slider("Sensor weight", 0.0, 1.0, 0.5, 0.05)
    image_w = 1.0 - sensor_w
    st.write(f"Image weight = {image_w:.2f}")

st.subheader("1) Upload Leaf Image")
img_file = st.file_uploader("Leaf image", type=["jpg","jpeg","png"])

st.subheader("2) Enter Sensor Readings")
c1,c2 = st.columns(2)
with c1:
    temp = st.number_input("Temperature (°C)", value=27.0)
    humid = st.number_input("Humidity (%)", value=85.0)
with c2:
    leaf_wet = st.number_input("Leaf Wetness (%)", value=72.0)
    soil_m = st.number_input("Soil Moisture (%)", value=62.0)

if st.button("Predict"):
    if img_file is None:
        st.warning("Please upload an image.")
    else:
        with st.spinner("Loading models and running inference..."):
            # load models
            sensor_model = load_sensor_model(sensor_model_path)
            cv_model_data = load_cv_model(cv_model_path, class_to_idx_path)

            # save image temp
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_file.read()); tmp.flush()
            sensors = {
                "temperature": temp,
                "humidity": humid,
                "leaf_wetness": leaf_wet,
                "soil_moisture": soil_m
            }
            result = combined_inference(sensor_model, cv_model_data, sensors, tmp.name, weights=Weights(sensor_weight=sensor_w, image_weight=image_w))
            #os.unlink(tmp.name)

        st.success("Done!")
        st.write("**Sensor-based risk**:", f"{result['sensor_prob']:.2f}")
        st.write("**Image-based risk**:", f"{result['image_prob']:.2f}")
        st.write("**Final risk (ensemble)**:", f"{result['final_prob']:.2f}")
        st.write("### Recommendations")
        for r in result["recommendations"]:
            st.write("- ", r)
