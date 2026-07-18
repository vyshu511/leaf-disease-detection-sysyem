# 🌿 LeafScope - AI Plant Disease Detection System

![Leaf Disease Detection](https://img.shields.io/badge/AI-Plant%20Disease%20Detection-green)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)

## 🚀 Live Demo

🌐 **Website:**  
https://leaf-disease-detection-sysyem.onrender.com

---

## 📌 Project Overview

LeafScope is an AI-powered plant disease detection system that identifies crop diseases from leaf images using a deep learning model.

Users can upload a leaf image and receive:

- 🌱 Disease prediction
- 📊 Confidence score
- 🔍 Top 3 possible predictions
- 🩺 Symptoms information
- 💊 Recommended treatments
- 🌿 Natural remedies
- 📜 Prediction history

The system is designed to help farmers and agriculture enthusiasts quickly identify plant diseases using computer vision.

---

## ✨ Features

### 🌿 AI Disease Detection
- Upload leaf images
- Automatic disease classification
- High-confidence predictions

### 🧠 Deep Learning Model
- MobileNetV2 Transfer Learning
- 12 plant disease classes
- Image classification using TensorFlow/Keras

### 📊 Prediction Insights
- Confidence percentage
- Top-3 predictions
- Disease severity estimation

### 🩺 Treatment Guidance
Provides:
- Symptoms
- Causes
- Chemical treatments
- Natural remedies

### 📜 History Tracking
- Stores previous predictions
- View recent scans

---

## 🏗️ System Architecture

```
User Uploads Leaf Image
          |
          ↓
Image Preprocessing
          |
          ↓
MobileNetV2 Deep Learning Model
          |
          ↓
Disease Prediction
          |
          ↓
Treatment Recommendation
```

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy

### Deployment
- Render
- Gunicorn

---

## 📂 Project Structure

```
leaf-disease-detection-sysyem/

│
├── app.py                 # Flask backend
├── train.py               # Model training script
├── requirements.txt       # Dependencies
├── runtime.txt            # Python version
│
├── model/
│   └── leaf_model.keras   # Trained MobileNetV2 model
│
├── utils/
│   └── gradcam.py         # Grad-CAM visualization
│
├── static/
│   ├── uploads/           # Uploaded images
│   ├── history/           # Prediction history
│
└── templates/
    └── index.html         # Frontend UI
```

---

## 🧠 Model Details

### MobileNetV2 Transfer Learning

- Pre-trained ImageNet model
- Fine-tuned for plant disease classification
- Input image size: 224 × 224 × 3

### Supported Classes

The model can detect 12 plant conditions including:

- Potato Early Blight
- Potato Late Blight
- Tomato Diseases
- Pepper Diseases
- Cabbage Diseases
- Soybean Healthy
- And more

---

## 📸 Application Screenshots

(Add screenshots here)

Example:

```
Upload Image → AI Prediction → Treatment Recommendation
```

---

## ⚙️ Installation & Running Locally

### Clone Repository

```bash
git clone https://github.com/vyshu511/leaf-disease-detection-sysyem.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🚀 Deployment

This project is deployed using Render.

Deployment configuration:

```
Build Command:
pip install -r requirements.txt

Start Command:
gunicorn app:app
```

Python Version:

```
python-3.11.9
```

---

## 🔮 Future Improvements

- 🌍 Multi-language farmer support
- 📱 Mobile application
- ☁️ Cloud database integration
- 🛰️ Satellite-based crop monitoring
- 🤖 AI chatbot for farming assistance
- 📈 Disease spread prediction

---

## 👩‍💻 Author

**Vyshnavi Manam**

AI & Data Science Student

GitHub:
https://github.com/vyshu511

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
