import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('model.h5')
labels = ['class_1', 'class_2', 'class_3']  # Example labels for classification

def predict_image(image_path):
    # Preprocess input image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize pixel values

    # Perform inference
    predictions = model.predict(x)

    # Get predicted label
    predicted_class = labels[np.argmax(predictions)]

    return predicted_class

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        predicted_label = predict_image(file_path)
        return jsonify({'prediction': predicted_label})

    return jsonify({'error': 'Prediction failed'})

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
