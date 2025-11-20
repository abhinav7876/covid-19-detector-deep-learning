# 🦠 **COVID-19 Chest X-Ray Classification (Deep Learning)**

This project uses a **Convolutional Neural Network (CNN)** to classify chest X-ray images as **Normal** or **COVID-19 Positive**. I collected datasets from GitHub (normal cases) and Kaggle (COVID cases), trained the model using **Google Colab GPU**, and deployed it using **Flask**.

---

## 📌 **Project Overview**

The goal of this project was to build an end-to-end deep learning pipeline:

* 📥 Collect raw X-ray images from two sources
* 🗂️ Prepare a clean dataset with proper train/test structure
* ⚡ Train a CNN using Colab GPU
* 💾 Save the trained model for inference
* 🌐 Deploy the model through a Flask web application

The final app allows users to upload a chest X-ray image and receive a prediction.

---

## 📁 **Dataset Details**

I used two publicly available datasets:

* 📄 **Normal chest X-rays** – GitHub repository
* 🩺 **COVID-19 positive X-rays** – Kaggle dataset

I cleaned and organized the images into the following structure:

```
final_dataset/
├── train/
│   ├── NORMAL/
│   └── COVID/
└── test/
    ├── NORMAL/
    └── COVID/
```

---

## ⚙️ **Model Training (Google Colab)**

Training was done in **Google Colab** to utilize GPU acceleration.

### 🧠 CNN Architecture Includes:

* Conv2D + MaxPooling layers
* Batch Normalization
* Dropout for regularization
* Dense layers for classification

I used `ImageDataGenerator` for augmentation (rotation, flipping, zooming, etc.).

Final model saved as:

```
covidmodel_final.h5
```

---

## 🚀 **Flask Deployment**

A simple Flask web application was created to serve the model. The app:

1. 📤 Accepts an uploaded X-ray image
2. 🖼️ Preprocesses it to the correct format
3. 🤖 Runs the model prediction
4. 📊 Returns **Normal** or **COVID-19 Positive**


---

## 🛠️ **How to Run the Project**

### 1️⃣ Clone the repository

```
git clone https://github.com/abhinav7876/Covid-19-Detector-Deep-Learning.git
```
### Create a conda environment after opening the repository

```
conda create -p venv python==3.10 -y
```

```
conda activate venv/
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Flask server

```
python app.py
```

---
![alt text](<Screenshot 2025-11-13 031955.png>)
