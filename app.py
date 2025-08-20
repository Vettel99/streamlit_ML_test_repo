import os
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

MODEL_ID = "14s_pVDNsO5laG0UI8c7SQK-sfGT97dmb"  # from your Drive link
MODEL_PATH = "FER_MobileNetV2_best.h5"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

# Download the model from Google Drive if not present
@st.cache_resource(show_spinner="Downloading model...")
def download_model():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(GDRIVE_URL)
            f.write(response.content)
    return load_model(MODEL_PATH)

# Load model (cached)
model = download_model()

# Emotion labels (example — replace with your actual label list)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Facial Emotion Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale if needed
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((128, 128))  # match model input size
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)  # adjust shape as needed

    prediction = model.predict(img_array)
    predicted_class = emotion_labels[np.argmax(prediction)]

    st.markdown(f"### Emotion: **{predicted_class}**")
