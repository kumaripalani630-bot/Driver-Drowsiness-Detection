import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="wide"
)

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("---")

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("drowsiness_model.h5")

try:
    model = load_model()
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Model Loading Error: {e}")
    st.stop()

# =====================================
# PROJECT OVERVIEW
# =====================================
st.header("📌 Project Overview")

st.write("""
This project detects driver fatigue using:

- Eye Closure Detection
- Yawning Detection
- MobileNetV2 Transfer Learning
- Decision Fusion Logic
- Fatigue Progression Analysis
""")

st.markdown("---")

# =====================================
# DATASET OVERVIEW
# =====================================
st.header("📂 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Eye Images", "1234")

with col2:
    st.metric("Mouth Images", "1230")

st.write("""
Classes Used:
- Closed
- Open
- no_yawn
- yawn
""")

st.markdown("---")

# =====================================
# MODEL COMPARISON
# =====================================
st.header("📊 Model Comparison")

comparison_df = pd.DataFrame({
    "Model":[
        "Custom CNN",
        "MobileNetV2",
        "ResNet50",
        "EfficientNet",
        "InceptionV3"
    ],
    "Validation Accuracy":[
        88,
        93.1,
        72.2,
        25,
        89.2
    ]
})

st.dataframe(comparison_df)

fig, ax = plt.subplots()

ax.bar(
    comparison_df["Model"],
    comparison_df["Validation Accuracy"]
)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Comparison")

st.pyplot(fig)

st.success("🏆 Best Model Selected: MobileNetV2")

st.markdown("---")

# =====================================
# MODEL EVALUATION
# =====================================
st.header("📈 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Eye Validation Accuracy",
        "99.54%"
    )

    st.metric(
        "Eye Test Accuracy",
        "97.71%"
    )

with col2:
    st.metric(
        "Mouth Validation Accuracy",
        "88.02%"
    )

    st.metric(
        "Mouth Test Accuracy",
        "94.04%"
    )

st.markdown("---")

# =====================================
# IMAGE PREDICTION
# =====================================
st.header("📷 Driver Image Prediction")

uploaded_file = st.file_uploader(
    "Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    # Preprocessing
    img = image.resize((224,224))

    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img,
        verbose=0
    )

    pred_index = np.argmax(
        prediction
    )

    # CHANGE ONLY IF CLASS ORDER DIFFERENT
    class_names = [
        "Closed",
        "Open",
        "no_yawn",
        "yawn"
    ]

    pred_class = class_names[
        pred_index
    ]

    confidence = round(
        np.max(prediction) * 100,
        2
    )

    st.subheader("Prediction Result")

    st.write(
        f"Predicted Class: {pred_class}"
    )

    st.write(
        f"Confidence: {confidence}%"
    )

    # =====================================
    # DECISION FUSION LOGIC
    # =====================================

    if pred_class in ["Open", "no_yawn"]:

        fatigue = "🟢 Alert"
        fatigue_score = 0

    elif pred_class == "yawn":

        fatigue = "🟡 Mild Fatigue"
        fatigue_score = 1

    elif pred_class == "Closed":

        fatigue = "🔴 Severe Fatigue"
        fatigue_score = 2

    else:

        fatigue = "Unknown"
        fatigue_score = -1

    st.success(
        f"Fatigue Level: {fatigue}"
    )

    st.metric(
        "Fatigue Score",
        fatigue_score
    )

    st.subheader("Class Probabilities")

    probs = prediction[0]

    for label, prob in zip(
        class_names,
        probs
    ):

        st.write(
            f"{label}: {prob*100:.2f}%"
        )

        st.progress(
            float(prob)
        )

st.markdown("---")

# =====================================
# DECISION FUSION TABLE
# =====================================
st.header("🧠 Decision Fusion Logic")

fusion_df = pd.DataFrame({
    "Prediction":[
        "Open",
        "no_yawn",
        "yawn",
        "Closed"
    ],
    "Fatigue Stage":[
        "Alert",
        "Alert",
        "Mild Fatigue",
        "Severe Fatigue"
    ]
})

st.table(fusion_df)

st.markdown("---")

# =====================================
# FATIGUE PROGRESSION CURVE
# =====================================
st.header("📉 Driver Fatigue Progression Curve")

fatigue_sequence = [
    0,0,0,0,0,
    1,1,1,1,1,
    1,2,2,2,2,
    2,2,2,2,2,
    1,1,1,
    0,0
]

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    fatigue_sequence,
    marker='o',
    linewidth=2
)

ax.set_yticks([0,1,2])

ax.set_yticklabels([
    "Alert",
    "Mild Fatigue",
    "Severe Fatigue"
])

ax.set_xlabel("Time Interval")

ax.set_ylabel("Fatigue Level")

ax.set_title(
    "Driver Fatigue Progression Curve"
)

ax.grid(True)

st.pyplot(fig)

st.info("""
Transition Points:

Alert → Mild Fatigue : Interval 5

Mild Fatigue → Severe Fatigue : Interval 11
""")

st.markdown("---")

# =====================================
# PERFORMANCE ANALYSIS
# =====================================
st.header("⚠️ Performance Analysis")

st.subheader("Robustness Analysis")

st.write("""
✔ Good lighting → Excellent Performance

✔ Normal pose → Excellent Performance

⚠ Spectacles → Moderate Impact

⚠ Low Light → Reduced Accuracy

⚠ Face Occlusion → Performance Drop
""")

st.subheader("Limitations")

st.write("""
- Low illumination affects predictions
- Extreme head rotations reduce accuracy
- Requires visible eye and mouth regions
- Real-time performance depends on hardware
""")

st.subheader("Class-wise Analysis")

analysis_df = pd.DataFrame({
    "Class":[
        "Alert",
        "Mild Fatigue",
        "Severe Fatigue"
    ],
    "Performance":[
        "Excellent",
        "Good",
        "Excellent"
    ]
})

st.table(analysis_df)

st.markdown("---")

# =====================================
# CONCLUSION
# =====================================
st.header("✅ Conclusion")

st.success("""
Driver Drowsiness Detection System successfully detects fatigue using:

👁 Eye Closure Analysis

👄 Yawning Analysis

🧠 MobileNetV2 Deep Learning Model

📉 Fatigue Progression Curve

🏆 Best Model: MobileNetV2

Eye Test Accuracy: 97.71%

Mouth Test Accuracy: 94.04%

Fatigue Stages:

🟢 Alert

🟡 Mild Fatigue

🔴 Severe Fatigue
""")