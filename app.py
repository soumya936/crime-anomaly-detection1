import streamlit as st
from PIL import Image
from model import extract_features, detect_anomaly

st.title("CCTV Human Action Anomaly Detection")

st.write("Upload CCTV frame to detect abnormal human activity")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = extract_features(image)

    result = detect_anomaly(features)

    st.subheader("Prediction Result")
    st.success(result)
