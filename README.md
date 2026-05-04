🧠 Driver Drowsiness Detection System using Deep Learning
📌 Project Description

The Driver Drowsiness Detection System is an AI-powered computer vision project designed to improve road safety by detecting early signs of driver fatigue. Fatigue-related accidents are a major concern in the transportation industry, often caused by reduced alertness, long driving hours, and lack of rest.

This system uses Deep Learning (CNN / MobileNetV2) and image-based facial analysis to identify driver drowsiness by monitoring two key physiological indicators:

👁 Eye state (Open / Closed)
😮 Yawning behavior (Yawning / Not Yawning)

Based on these inputs, the system classifies the driver’s condition into different fatigue levels and provides real-time feedback.

A Streamlit-based web application is developed to provide an interactive interface for image upload, dataset visualization, prediction results, and fatigue progression simulation.

🎯 Problem Statement

Driver fatigue significantly reduces reaction time, awareness, and decision-making ability, leading to severe road accidents. Traditional methods such as vehicle-based sensors or wearable devices are often intrusive or unreliable.

There is a need for a non-intrusive, vision-based intelligent system that can detect driver drowsiness using facial behavior analysis in real-time.

🧠 Proposed Solution

This project proposes an AI-based solution that:

Captures or receives driver facial images
Analyzes eye closure and yawning patterns
Uses a trained deep learning model to classify states
Maps predictions into fatigue levels
Displays results in an interactive dashboard

The system is designed to be lightweight, scalable, and suitable for integration into Advanced Driver Assistance Systems (ADAS).

⚙️ System Architecture
1. Data Collection
Facial images categorized into:
Eyes Open
Eyes Closed
Yawn
No Yawn
2. Data Preprocessing
Image resizing (224 × 224)
Normalization (0–1 scaling)
Data augmentation:
Rotation
Zoom
Horizontal flipping
Brightness adjustment
3. Model Development

Two approaches are considered:

Custom Convolutional Neural Network (CNN)
Transfer Learning using MobileNetV2

These models are trained to extract facial features and classify eye and mouth states.

4. Decision Fusion Logic

The final fatigue level is derived using rule-based logic:

🟢 Alert → Eyes Open + No Yawn
🟡 Mild Fatigue → Yawning detected
🔴 Severe Fatigue → Eyes Closed
5. Deployment

The trained model is deployed using Streamlit, providing:

Image upload interface
Real-time prediction display
Dataset preview (from ZIP file)
Fatigue progression simulation graph
📊 Key Features
👁 Eye state detection
😮 Yawning detection
🧠 Deep learning-based classification
📦 Dataset handling via ZIP extraction
📷 Image upload and prediction system
📊 Fatigue progression visualization
🌐 Interactive Streamlit web dashboard
📌 Business Applications
🚗 Accident prevention systems
🚛 Fleet management safety monitoring
🛣️ Smart transportation systems
🧾 Insurance risk analysis
🤖 ADAS (Advanced Driver Assistance Systems) integration
📊 Output Classes
Class	Description
🟢 Alert	Driver is fully attentive
🟡 Mild Fatigue	Early signs of drowsiness
🔴 Severe Fatigue	High risk of accident
🛠️ Technologies Used
Python 🐍
TensorFlow / Keras 🤖
Convolutional Neural Networks (CNN)
MobileNetV2 (Transfer Learning)
OpenCV 👁
Streamlit 🌐
Matplotlib 📊
NumPy
📂 Dataset Information

The dataset consists of labeled facial images used for training and evaluation:

Eye state classification (Open / Closed)
Mouth state classification (Yawn / No Yawn)

The dataset is preprocessed and organized into training, validation, and testing sets for deep learning model development.

📈 Results & Performance
High accuracy in eye and mouth classification tasks
Robust performance under varying lighting conditions
Effective fatigue detection using fusion logic
Clear visualization of fatigue progression trends
🚀 Future Enhancements
Real-time webcam-based detection
Audio alert system for drowsy drivers
Integration with vehicle sensors (ADAS systems)
Mobile application deployment
Edge AI implementation (Raspberry Pi / Jetson Nano)
Cloud-based fleet monitoring dashboard
👨‍💻 Author

Your Name
AI & Data Science Student

📌 Conclusion

This project demonstrates how Artificial Intelligence and Computer Vision can be applied in real-world transportation systems to enhance safety. By detecting early signs of driver fatigue, the system helps reduce accident risks and supports the development of intelligent driving assistance technologies.
