import sys
import os
from PIL import Image
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.predict import predict_leaf

st.set_page_config(page_title="Plant Leaf Disease Prediction", layout="centered")

st.title("Plant Leaf Disease Prediction System")
st.write("Upload a plant leaf image to predict plant name, health status, disease, and cure.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", width="stretch")

    with st.spinner("Predicting..."):
        result = predict_leaf(image)

    if result["warning_level"] == "medium":
        st.warning(
            "Prediction confidence is moderate. The image may be unclear or from a difficult real-world condition."
        )

    if not result["is_supported"]:
        st.warning(
            "This image may belong to an unsupported plant or may be unclear. "
            "Please upload a leaf from a supported plant: Tomato, Potato, or Pepper."
        )
        st.stop()

    st.markdown("### Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Plant Name", result["plant_name"])
        st.metric("Plant Type", result["plant_type"])
        st.metric("Leaf Status", result["leaf_status"])

    with col2:
        st.metric("Disease", result["disease_name"])
        st.metric("Confidence", f"{result['confidence']}%")

    st.markdown("### Suggested Cure")
    st.info(result["cure"])