from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os

app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load tokenizer and models once
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
caption_model = load_model('model.keras')
feature_extractor = load_model('feature_extractor.keras')

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
