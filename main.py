from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import torch
from dataset import transform
from flask import render_template
import sqlite3
from similarity import load_model, get_top_similar
from dataset import animal_array
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('trained_model.pth', device)

conn = sqlite3.connect('features.db', check_same_thread=False)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features, predictions = model(image)

        features = features.cpu().numpy()
        similarities = get_top_similar(features, conn, top_n=5, threshold=0.7)
        if not similarities:
            return jsonify({'error': 'No matches found'}), 201
        
        res = []
        for _, b, _ in similarities:
            res.append(b)

        return jsonify({
            'predicted_class': str(animal_array[torch.argmax(predictions)]),
            'similar_images': res
        })


if __name__ == '__main__':
    app.run(debug=True)
