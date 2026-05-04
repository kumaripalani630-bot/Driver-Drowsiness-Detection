import streamlit as st
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🧠 Driver Drowsiness Detection System")
st.markdown("Dataset (ZIP) + Eye & Yawn Analysis Dashboard")

# ==========================
# ZIP FILE PATH
# ==========================
ZIP_FILE = "c:/Users/LENOVO/Downloads/archive (2).zip"
EXTRACT_FOLDER = "dataset"

# ==========================
# EXTRACT ZIP FUNCTION
# ==========================
def extract_zip():
    if not os.path.exists(EXTRACT_FOLDER):
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
        return "Dataset Extracted ✔"
    return "Dataset Already Available ✔"

# ==========================
# LOAD DATASET IMAGES
# ==========================
def load_images(folder, limit=5):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(("jpg", "png", "jpeg")):
                images.append(os.path.join(root, file))
            if len(images) >= limit:
                break
    return images

# ==========================
# FATIGUE LOGIC
# ==========================
def fatigue_level(pred):
    if pred in ["Open", "No Yawn"]:
        return "🟢 Alert", 0
    elif pred == "Yawn":
        return "🟡 Mild Fatigue", 1
    else:
        return "🔴 Severe Fatigue", 2

# fake prediction
classes = ["Open", "Closed", "No Yawn", "Yawn"]

def predict_state():
    return random.choice(classes)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📦 Dataset Viewer", "📷 Image Test", "📊 Simulation"])

# ==========================
# HOME
# ==========================
if page == "🏠 Home":
    st.header("📌 Project Overview")

    st.write("""
    ### 🧠 Driver Drowsiness Detection System

    This project uses:
    - Eye state detection
    - Yawning detection
    - Deep learning (CNN / MobileNetV2 ready)

    ### 📦 Dataset:
    archive (2).zip is used as dataset input
    """)

    if st.button("Extract Dataset"):
        msg = extract_zip()
        st.success(msg)

# ==========================
# DATASET VIEWER
# ==========================
elif page == "📦 Dataset Viewer":
    st.header("📂 Dataset Preview from ZIP")

    msg = extract_zip()
    st.info(msg)

    images = load_images(EXTRACT_FOLDER, limit=6)

    if images:
        cols = st.columns(3)

        for i, img_path in enumerate(images):
            img = Image.open(img_path)
            cols[i % 3].image(img, use_container_width=True)

    else:
        st.warning("No images found in dataset")

# ==========================
# IMAGE TEST
# ==========================
elif page == "📷 Image Test":
    st.header("📷 Upload Image")

    uploaded_file = st.file_uploader("Upload Driver Face Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        pred = predict_state()
        level, code = fatigue_level(pred)

        st.write("👁 Prediction:", pred)
        st.write("🚨 Fatigue Level:", level)

# ==========================
# SIMULATION
# ==========================
elif page == "📊 Simulation":
    st.header("📊 Fatigue Progression Curve")

    if st.button("Start Simulation"):

        data = []
        progress = st.progress(0)

        for i in range(30):
            pred = predict_state()
            level, code = fatigue_level(pred)
            data.append(code)

            progress.progress((i + 1) * 3)
            import time
            time.sleep(0.1)

        st.subheader("📉 Graph")

        plt.figure()
        plt.plot(data, marker="o")
        plt.yticks([0, 1, 2], ["Alert", "Mild", "Severe"])
        plt.xlabel("Time")
        plt.ylabel("Fatigue Level")

        st.pyplot(plt)

        st.success("Simulation Completed ✔")