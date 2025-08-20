import os
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import gdown  

MODEL_PATH = "FER_MobileNetV2_best.h5"
MODEL_ID = "14s_pVDNsO5laG0UI8c7SQK-sfGT97dmb"
GDRIVE_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

@st.cache_resource(show_spinner="Downloading model from Google Drive...")
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_model_from_drive()

# Your app logic follows


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
