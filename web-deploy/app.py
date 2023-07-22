from __future__ import division, print_function

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from os.path import join, dirname, realpath

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.utils import img_to_array, load_img

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "C:/Users/dkvhe/OneDrive/Documents/vs_code/Projects/ML/web-deploy/models/pancreatic_tumor_model.h5"

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('')
print("Model loaded. Check http://127.0.0.1:5000/")


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150))
    input_arr = img_to_array(img) / 225
    input_arr = np.expand_dims(input_arr, axis=0)
    val = model.predict(input_arr)
    pred = (model.predict(input_arr) > 0.5).astype("int32")
    print(val)
    if pred == 0:
        result = "image is normal" + "(Accuracy: " + str(val) + " )"
    else:
        result = "image is having a Tumor" + "(Accuracy: " + str(val) + " )"
    return result


@app.route("/", methods=["GET"])
def index():
    # Main page
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        UPLOADS_PATH = join(dirname(realpath(__file__)), "static\\image.jpg")

        f.save(UPLOADS_PATH)

        # Make prediction
        preds = model_predict(UPLOADS_PATH, model)

        return preds
    return None


if __name__ == "__main__":
    app.run(debug=True)
