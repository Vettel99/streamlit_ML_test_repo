import os
import io
import requests
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model  # <-- tf.keras (NOT keras 3)

# ====== CONFIG ======
MODEL_URL  = "https://huggingface.co/Vettel99/FER_MobileNetV2_best/resolve/main/FER_MobileNetV2_best.h5"
MODEL_PATH = "FER_MobileNetV2_best.h5"

DEFAULT_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# ====== UTILITIES ======
def _is_valid_hdf5(path: str) -> bool:
    """Quick magic-bytes check to ensure it's a real HDF5 file."""
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def _download_file(url: str, dest_path: str) -> None:
    """Stream the model file from HF to disk safely."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
        # Basic sanity check on size if header present
        if total and written < max(1024, int(0.95 * total)):
            raise RuntimeError("Downloaded file size is suspiciously small; aborting.")


@st.cache_resource(show_spinner="Downloading & loading model…")
def load_emotion_model():
    # Fresh download if missing or invalid
    if (not os.path.exists(MODEL_PATH)) or (not _is_valid_hdf5(MODEL_PATH)):
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except OSError:
                pass
        _download_file(MODEL_URL, MODEL_PATH)
        if not _is_valid_hdf5(MODEL_PATH):
            raise RuntimeError("Downloaded file is not a valid HDF5 (HDF5 signature not found).")

    # Use tf.keras loader; compile=False avoids optimizer/metric deserialization issues
    model = load_model(MODEL_PATH, compile=False)
    return model


def _prep_image(img: Image.Image, input_shape):
    """
    Preprocess PIL image to match model.input_shape (None, H, W, C).
    - Converts to RGB or L as required
    - Resizes to (W, H)
    - Scales to [0,1]
    """
    # Normalize shape: Keras gives (None, H, W, C)
    if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
        # If the model has multiple inputs, pick the first (rare for this use-case)
        ishape = input_shape[0]
    else:
        ishape = input_shape

    _, H, W, C = ishape  # channel-last expected

    # Color mode per channels
    if C == 1:
        img = img.convert("L")
    else:
        # If model expects 3 channels, ensure RGB
        img = img.convert("RGB")

    # Resize (PIL expects (width, height))
    img = img.resize((W, H), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if C == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
    arr = np.expand_dims(arr, axis=0)       # (1, H, W, C)
    return arr


# ====== APP ======
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("Facial Emotion Recognition (MobileNetV2)")

# Load model (with friendly failure)
try:
    model = load_emotion_model()
except Exception as e:
    st.error(
        "❌ Failed to load the model. "
        "Ensure the Space uses TensorFlow 2.x and the URL is reachable.\n\n"
        f"Details: {type(e).__name__}: {e}"
    )
    st.stop()

# Derive labels robustly
num_out = int(model.output_shape[-1])
labels = DEFAULT_LABELS if len(DEFAULT_LABELS) == num_out else [f"Class {i}" for i in range(num_out)]

# Show model info
with st.expander("Model details", expanded=False):
    st.write(f"**Input shape:** `{model.input_shape}`  |  **Output classes:** `{num_out}`")
    st.write(f"**Labels used:** {labels}")

uploaded = st.file_uploader("Upload a face image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        img = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess based on model’s true input shape
    x = _prep_image(img, model.input_shape)

    # Inference
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=-1)[0])
    conf = float(np.max(preds, axis=-1)[0])

    st.subheader(f"Prediction: **{labels[idx]}**")
    st.caption(f"Confidence: {conf:.3f}")

    # Optional: simple probabilities table
    show_probs = st.checkbox("Show class probabilities", value=False)
    if show_probs:
        for i, p in enumerate(preds[0].tolist()):
            st.write(f"{labels[i]}: {p:.4f}")

st.markdown("---")
st.caption("Tip: If you trained with a different image size or color mode, this app adapts to the model’s input shape automatically.")
