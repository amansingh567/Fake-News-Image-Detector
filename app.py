import streamlit as st
import re
import nltk
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Load models
loaded_model, tfidf_v = joblib.load("models/text_classifier.pkl")
model_image = tf.keras.models.load_model("models/image_cnn_model.keras")

st.title("📰 Fake News & Image Validation System")

news = st.text_area("Paste the news article text here:")
img_file = st.file_uploader("Upload an associated image (optional):", type=["jpg", "png"])

if st.button("🔍 Analyze"):
    if news:
        lemmatizer = WordNetLemmatizer()
        stpwrds = list(stopwords.words('english'))
        review = news
        review = re.sub(r'[^a-zA-Z\s]', ' ', review)
        review = review.lower()
        review = nltk.word_tokenize(review)
        cleaned = [lemmatizer.lemmatize(word) for word in review if word not in stpwrds]     
        input_data = [' '.join(cleaned)]

        vectorized_input_data = tfidf_v.transform(input_data)
        prediction = loaded_model.predict(vectorized_input_data)
        decision_scores = loaded_model.decision_function(vectorized_input_data)
        # decision_scores is an array like [value]

        # Convert to confidence
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        prob_fake = sigmoid(decision_scores[0]) 
        prob_real = 1 - prob_fake


        if prediction[0] == 1:
            st.warning("📰 Prediction of the News :  Looking Fake⚠ News (Confidence: %.2f)" %prob_fake)
        else:
            st.success("📰 Prediction of the News : Looking Real News (Confidence: %.2f) "%prob_real)

            
        
    if img_file:
        test_image = image.load_img(img_file, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        result = model_image.predict(test_image)[0][0]
        if result < 0.5:
            st.warning(f"🖼️ Image Prediction: TAMPERED (Confidence: {1 - result:.2f})")
        else:
            st.success(f"🖼️ Image Prediction: GENUINE (Confidence: {result:.2f})")