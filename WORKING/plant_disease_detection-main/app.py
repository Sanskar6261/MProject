import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')
# model = tf.keras.models.load_model('model.h5')
# model = tf.keras.models.load_model('model.h5')
print('Model loaded successfully.')
print(model.summary())

print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def get_result(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(225, 225))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            predictions = get_result(file_path)
            predicted_label = labels[np.argmax(predictions)]
            return predicted_label
    return "Error: No file provided."

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
 