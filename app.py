import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

st.title("CCTV Anomaly Detection System")

st.write("Upload an image to detect anomaly")

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:

    from PIL import Image
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing...")

    # Example output
    prediction = np.random.choice(["Normal Activity","Anomalous Activity"])

    st.success(f"Prediction: {prediction}")
