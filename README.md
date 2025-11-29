# ♻️ RecycleVision: Waste Classification with Deep Precision

RecycleVision is a deep learning–based image classification system that detects and classifies garbage into six categories using transfer learning and convolutional neural networks.  
The goal is to enable **smart, automated waste segregation** using Artificial Intelligence.

---

## 📌 Features

✅ Classifies images into 6 waste categories  
✅ Transfer learning using MobileNetV2  
✅ Fine-tuned deep learning model  
✅ Streamlit-based web interface  
✅ Real-time predictions  
✅ Accuracy & loss visualization  
✅ Confusion matrix and classification report  
✅ Lightweight and deployable model  

---

## 🗂️ Waste Categories

The model classifies images into the following categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## 📊 Model Performance

- Training Accuracy: **96%**
- Validation Accuracy: **~80%**
- Model type: Transfer Learning (MobileNetV2)
- Optimizer: Adam
- Epochs: 25

---

## 🏗️ Project Structure

      RecycleVision/
      │
      ├── dataset/
      │ ├── train/
      │ └── test/
      │
      ├── train.py
      ├── evaluation.py
      ├── eda.py
      ├── app.py
      │
      ├── accuracy_plot.png
      ├── loss_plot.png
      ├── class_distribution.png
      │
      └── garbage_classifier.h5

## 🖥️ Application Preview

The web application allows users to upload a garbage image and instantly receive:

-Predicted waste category

-Confidence score

## 🎯 Use Cases

✅Smart recycling systems

✅Waste sorting automation

✅Educational environments

✅Environmental awareness systems

✅Smart cities projects

## 📚 Technologies Used

-Python

-TensorFlow 

-Keras

-MobileNetV2

-Scikit-learn

-OpenCV

-Streamlit

-Matplotlib





