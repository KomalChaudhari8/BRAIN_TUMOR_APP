🧠Brain Tumor Classification Using CNN and MobileNetV2

This project implements a deep learning-based brain tumor classifier that detects four types of brain tumors (glioma, meningioma, notumor, and pituitary) using MRI images. 
It utilizes a custom Convolutional Neural Network (CNN) model and a MobileNetV2-based transfer learning approach.

📂 Dataset
- Source: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Classes: `glioma`, `meningioma`, `notumor`, `pituitary`

 🚀 Features

- ✅ Custom CNN using MobileNetV2
- 🎯 Improved class balance using Focal Loss
- 📈 Achieved over 90% accuracy on the test set
- 🌐 Flask web app for easy deployment and prediction
- 📊 Performance metrics: confusion matrix, classification report
- 🗂 Kaggle dataset support

 🛠️ Tech Stack

- Framework: TensorFlow / Keras
- Frontend: Flask (for deployment)
- Languages: Python
- Others: NumPy, Matplotlib, Seaborn

📝 How to Run Locally

Clone the repository: 
git clone https://github.com/KomalChaudhari8/BRAIN_TUMOR_APP.git 
cd brain-tumor-classifier

Install dependencies: 
pip install -r requirements.txt

Run the Flask app: 
python app.py

Open http://127.0.0.1:5000 in your browser.
