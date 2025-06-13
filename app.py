import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
import json
import os
import gdown

# Set up page
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# 📥 Load model from Google Drive
MODEL_URL = 'https://drive.google.com/uc?id=1lDkbTBf215mCxONJHJ02OhG4sSmJOCTK'
MODEL_PATH = 'plant_disease_detector_vgg19.keras'

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load trained model
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

# Load class labels
try:
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
except FileNotFoundError:
    st.error("Missing class_labels.json. Please generate it from your training code.")
    st.stop()

# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to identify the disease.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(256, 256))
    img_array = img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    # Prediction
    prediction = model.predict(img_expanded)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.success(f"🩺 Prediction: **{class_labels[str(predicted_class)]}**")
    st.info(f"Confidence: {confidence:.2%}")


#To Run: python -m streamlit run app.py