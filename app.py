import os
import requests
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

MODEL_URL = "https://huggingface.co/Vettel99/FER_MobileNetV2_best/resolve/main/FER_MobileNetV2_best.h5"
MODEL_PATH = "FER_MobileNetV2_best.h5"

@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(MODEL_URL)
            if response.status_code != 200:
                raise Exception("Failed to download model file.")
            f.write(response.content)
    return load_model(MODEL_PATH)

model = load_emotion_model()

# Replace with your actual label order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Facial Emotion Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)

    prediction = model.predict(img_array)
    predicted_label = emotion_labels[np.argmax(prediction)]

    st.markdown(f"### Emotion Detected: **{predicted_label}**")

