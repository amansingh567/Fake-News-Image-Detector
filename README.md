# Fake News & Image Validation System

This project is a web application that detects fake news articles and validates associated images using machine learning and deep learning models. It combines Natural Language Processing (NLP) for text classification and Convolutional Neural Networks (CNN) for image authenticity detection.

---

## Features

- **Fake News Detection:** Classifies news text as either *fake* or *real* using a TF-IDF vectorizer and a Passive Aggressive Classifier.
- **Image Validation:** Analyzes uploaded images to predict if they are genuine or tampered using a CNN model.
- **Interactive Web UI:** Built using Streamlit for ease of use and quick deployment.

---

## Demo

You can run this app locally or deploy it on platforms like Render or Streamlit Cloud.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. (Optional but recommended) Create and activate a virtual environment:
  python -m venv .venv
    # On Windows
      .venv\Scripts\activate
    # On macOS/Linux
      source .venv/bin/activate

3. Install dependencies:
  pip install -r requirements.txt

4. Project Structure
   
  ├── models/
  
  │   ├── text_classifier.pkl          # Pretrained text classification model
  │   └── image_cnn_model.keras         # Pretrained image CNN model
  
  ├── datasets/                        # (Optional) datasets used for training
  
  ├── app.py                          # Main Streamlit app code
  
  ├── requirements.txt                # Python dependencies
  
  └── README.md                      # This file

  5.Usage
    Run the Streamlit app:
        streamlit run app.py
        
    Paste or type the news article text in the input box.
    Upload an image file (optional).
    Click Analyze to see predictions for news authenticity and image validation.
