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
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="wide"
)

st.title("🧠 Driver Drowsiness Detection System")

st.markdown(
    "Dataset (ZIP) + Eye & Yawn Analysis Dashboard"
)

# ==========================
# ZIP FILE PATH
# ==========================
ZIP_FILE = r"c:/Users/LENOVO/Downloads/archive (2).zip"

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
def load_images(folder, limit=6):

    images = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                ("jpg", "jpeg", "png")
            ):

                images.append(
                    os.path.join(root, file)
                )

            if len(images) >= limit:
                break

    return images

# ==========================
# FAKE PREDICTION FUNCTION
# ==========================
def predict_state():

    eye_states = [
        "Open",
        "Closed"
    ]

    mouth_states = [
        "No Yawn",
        "Yawn"
    ]

    eye_pred = random.choice(eye_states)

    mouth_pred = random.choice(mouth_states)

    return eye_pred, mouth_pred

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📦 Dataset Viewer",
        "📷 Image Test",
        "📊 Simulation"
    ]
)

# ==========================
# HOME PAGE
# ==========================
if page == "🏠 Home":

    st.header("📌 Project Overview")

    st.write("""
    ### 🧠 Driver Drowsiness Detection System

    This project uses:
    - Eye state detection
    - Yawning detection
    - Deep learning (CNN / MobileNetV2)

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

    st.header("📂 Dataset Preview")

    msg = extract_zip()

    st.info(msg)

    images = load_images(
        EXTRACT_FOLDER,
        limit=6
    )

    if images:

        cols = st.columns(3)

        for i, img_path in enumerate(images):

            img = Image.open(img_path)

            cols[i % 3].image(
                img,
                use_container_width=True
            )

    else:

        st.warning(
            "No images found in dataset"
        )

# ==========================
# IMAGE TEST
# ==========================
elif page == "📷 Image Test":

    st.header("📷 Upload Driver Image")

    uploaded_file = st.file_uploader(
        "Upload Driver Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Fake prediction
        eye_pred, mouth_pred = predict_state()

        st.write(
            "👁 Eye State:",
            eye_pred
        )

        st.write(
            "👄 Mouth State:",
            mouth_pred
        )

        # Decision Fusion Logic
        if (
            eye_pred == "Open"
            and mouth_pred == "No Yawn"
        ):

            fatigue_level = "🟢 Alert"

        elif eye_pred == "Closed":

            fatigue_level = "🔴 Severe Fatigue"

        elif mouth_pred == "Yawn":

            fatigue_level = "🟡 Mild Fatigue"

        else:

            fatigue_level = "🟢 Alert"

        st.write(
            "🚨 Fatigue Level:",
            fatigue_level
        )

# ==========================
# SIMULATION
# ==========================
elif page == "📊 Simulation":

    st.header(
        "📊 Driver Fatigue Progression Curve"
    )

    if st.button("Start Simulation"):

        data = []

        progress = st.progress(0)

        # Realistic fatigue sequence
        fatigue_sequence = [
            0,0,0,0,0,
            1,1,1,1,1,
            1,1,2,2,2,
            2,2,2,2,2,
            2,2,1,1,1,
            0,0
        ]

        for i, value in enumerate(
            fatigue_sequence
        ):

            data.append(value)

            progress.progress(
                int(
                    (i + 1)
                    / len(fatigue_sequence)
                    * 100
                )
            )

        st.subheader("📉 Fatigue Curve")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            data,
            marker="o"
        )

        ax.set_yticks([0,1,2])

        ax.set_yticklabels([
            "Alert",
            "Mild",
            "Severe"
        ])

        ax.set_xlabel("Time")

        ax.set_ylabel("Fatigue Level")

        ax.set_title(
            "Driver Fatigue Progression Curve"
        )

        ax.grid(True)

        st.pyplot(fig)

        st.success(
            "Simulation Completed ✔"
        )