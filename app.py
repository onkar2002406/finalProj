from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os

app = Flask(__name__)

import os
import gdown  # to download from Google Drive

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# URLs from your uploaded files (replace with your links)
TOKENIZER_URL = 'https://drive.google.com/file/d/1CMTOfRjImcWAQU5jCXnZ41OvvER-HYLt/view?usp=drive_link'
MODEL_URL = ' https://drive.google.com/file/d/16qeaFL_r6jjEw2VCZDp2uszAPEI8NH7v/view?usp=drive_link'
FEATURE_EXTRACTOR_URL = 'https://drive.google.com/file/d/1mq2w5HFpRXAonDXl6H2wVOV8aSmbiqyz/view?usp=drive_link'

def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Auto-download models
download_file(TOKENIZER_URL, os.path.join(MODEL_DIR, 'tokenizer.pkl'))
download_file(MODEL_URL, os.path.join(MODEL_DIR, 'model.keras'))
download_file(FEATURE_EXTRACTOR_URL, os.path.join(MODEL_DIR, 'feature_extractor.keras'))

# Now, load the models
import pickle
from tensorflow.keras.models import load_model

tokenizer = pickle.load(open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb'))
caption_model = load_model(os.path.join(MODEL_DIR, 'model.keras'))
feature_extractor = load_model(os.path.join(MODEL_DIR, 'feature_extractor.keras'))


max_length = 34

def preprocess_image(image_path, size=224):
    img = load_img(image_path, target_size=(size, size))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def extract_features(image):
    return feature_extractor.predict(image, verbose=0)

def generate_caption(image_features):
    caption = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Proper input shapes for model
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        idx = np.argmax(yhat)
        word = tokenizer.index_word.get(idx, None)

        if word is None or word == 'endseq':
            break
        caption += ' ' + word

    return caption.replace('startseq', '').replace('endseq', '').strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify(status='error', message='No file uploaded!')

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    processed_image = preprocess_image(image_path)
    features = extract_features(processed_image)

    caption = generate_caption(features)

    return jsonify(status='success', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
