# 📰🖼️ Fake News & Image Validation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-yellow?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

A web application that detects **fake news articles** and validates **associated images** using Machine Learning and Deep Learning.  
This project integrates:
- 🧠 **Natural Language Processing (NLP)** for fake news detection  
- 🖼️ **Convolutional Neural Networks (CNN)** for image tampering detection

---

## 🎥 Demo

📽️ **Video Demo**:  
[Click here to watch](https://drive.google.com/file/d/1iWx1yGh_3HST9KluA5wMDEXGy5WGrzF0/view?usp=drive_link)
 

---

## 🚀 Features

- ✅ **Fake News Detection**  
  Classifies input text as **Real** or **Fake** using a TF-IDF Vectorizer and Passive Aggressive Classifier.

- 🖼️ **Image Validation**  
  Analyzes uploaded images to determine if they are **Genuine** or **Tampered**, using a custom-trained CNN.

- 💻 **Interactive Web Interface**  
  Built with **Streamlit** for real-time predictions and a user-friendly UI.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. (Optional) Set Up Virtual Environment
On Windows
```
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Dependencies
```
pip install -r requirements.txt
```

📁 Project Structure
.
├── models/
│   ├── text_classifier.pkl         # Pretrained text classification model
│   └── image_cnn_model.keras       # Pretrained image tampering detection model
├── datasets/                       # (Optional) Training datasets
├── app.py                          # Main Streamlit app file
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation


🚦 Usage
Run the Streamlit app locally with:
```
streamlit run app.py
```
Then:

Paste or type a news article in the input box.
(Optional) Upload an image to verify authenticity.
Click Analyze to view both predictions:
Fake News Prediction

Image Validation Result

🧠 Technologies Used
Python 3.9+
NLTK – Tokenization, stopword removal, lemmatization
Scikit-learn – TF-IDF vectorization, Passive Aggressive Classifier
TensorFlow/Keras – CNN-based image validation
Streamlit – Frontend web interface
OpenCV – Image manipulation

📌 Notes
Ensure the following model files are present in the models/ folder:

text_classifier.pkl
image_cnn_model.keras

You can retrain the models using your own datasets from the datasets/ directory.

🌱 Future Improvements

✅ Deploy on Render / Streamlit Cloud with CI/CD
✅ Add logging and error handling

📜 License
This project is licensed under the MIT License.








