<h1 align="center">🧠 Handwritten Digit Classification</h1>

<p align="center">
  <b>Deep Learning-powered Digit Recognition using CNN 🔢</b><br/>
  <i>Upload an image → Get instant prediction (0–9)</i>
</p>

<p align="center">
  <a href="https://digitclassificationmodel-vtqft8zyinghfkscftholz.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-blue?style=for-the-badge&logo=streamlit"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-DeepLearning-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-CNN-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-MNIST-blueviolet?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

<div align="center">

🧠 This project classifies **handwritten digits (0–9)** using a
**Convolutional Neural Network (CNN)** trained on the MNIST dataset.

📷 Users can upload an image → Model predicts the digit instantly.

</div>

---

## 🎯 Objective

* Recognize handwritten digits accurately
* Demonstrate CNN-based image classification
* Provide an interactive web interface
* Deploy model using Streamlit

---

## ✨ Key Features

<ul>
  <li>🧠 Deep Learning (CNN Model)</li>
  <li>📷 Image Upload & Prediction</li>
  <li>⚡ Real-time Classification</li>
  <li>🌐 Web App using Streamlit</li>
  <li>📊 High accuracy on MNIST dataset</li>
</ul>

---

## 🧠 Model Architecture

<div style="background:#0d1117; padding:15px; border-radius:10px; border:1px solid #30363d;">

📌 Input Layer (28x28 grayscale image)
➡️ Convolution Layers
➡️ Activation (ReLU)
➡️ Pooling Layers
➡️ Fully Connected Layers
➡️ Output Layer (10 classes: digits 0–9)

</div>

---

## 🏗️ Tech Stack

<table align="center">
<tr><td><b>Language</b></td><td>Python</td></tr>
<tr><td><b>Libraries</b></td><td>PyTorch, NumPy, Pillow</td></tr>
<tr><td><b>Framework</b></td><td>Streamlit</td></tr>
<tr><td><b>Model File</b></td><td>.pth (PyTorch model)</td></tr>
</table>

---

## 📂 Project Structure

<div style="background:#0d1117; padding:15px; border-radius:10px; border:1px solid #30363d; font-family:monospace;">

Digit_classification_model/ <br/>├── app.py <br/>├── digit_cnn_model.pth <br/>├── requirements.txt <br/>└── README.md

</div>

---

## ⚙️ Installation & Setup

```bash id="m8v5r7"
git clone https://github.com/Yamuna-97/Digit_classification_model
cd Digit_classification_model
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 How It Works

1. Upload a handwritten digit image
2. Image is preprocessed (resized, normalized)
3. CNN model analyzes the image
4. Prediction result (0–9) is displayed

---

## 📊 Model Insights

* Trained on **MNIST dataset**
* Handles grayscale images (28x28)
* Learns spatial features using convolution layers
* Provides fast and accurate predictions

---

## 🌐 Live Demo

<p align="center">
  <a href="https://digitclassificationmodel-vtqft8zyinghfkscftholz.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Try%20Now-Launch%20App-blue?style=for-the-badge&logo=streamlit"/>
  </a>
</p>

---

## 📊 Highlights

✔ Deep Learning Implementation

✔ CNN-based Image Classification

✔ Real-time Prediction

✔ Deployed Web Application

---

## 🔮 Future Enhancements

* 🔍 Add probability scores for predictions
* 📊 Visualization of model predictions
* 🧠 Improve accuracy with advanced CNN architectures
* ☁️ Deploy using cloud services

---

<div align="center">
✨ “Teaching Machines to See Numbers.” ✨
</div>
