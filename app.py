import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flask import Flask,render_template,request
import pickle
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import load_model
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flask import Flask, render_template, request
import cv2        
import tensorflow as tf
import numpy as np
from keras.models import load_model

app = Flask(__name__)
density_constant = 0.011219234
model = load_model('model(1).h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image from the request
        image_file = request.files['image']

        if image_file:
            # Read the image using OpenCV
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Resize the image to match the model's input size
            input_image = cv2.resize(image, (256, 256))
            input_image = input_image / 255.0

            # Make predictions using your ML model
            predictions = model.predict(tf.expand_dims(input_image, axis=0))
            total_dens_pred = tf.reduce_sum(predictions, axis=[1, 2, 3]).numpy()

            return "Number of trees in given input is: " + str(total_dens_pred[0] * density_constant)

if __name__ == '__main__':
    app.run(debug=True)
