from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow
tensorflow.compat.v1.disable_eager_execution()
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow import Graph
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template


app = Flask(__name__)


from tensorflow.python.keras.backend import set_session
sess = tensorflow.compat.v1.Session()
graph=tensorflow.compat.v1.get_default_graph()


MODEL_PATH = 'covidmodel.h5'

set_session(sess)
model = load_model(MODEL_PATH)
model._make_predict_function()  # Necessary

print('Model loaded. Start serving...')




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x=x/255.0
    y=x.reshape((224,224,3))
    y = np.expand_dims(y, axis=0)
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        preds = model.predict(y)
    if(preds[0][0]<0.5):
        result="Covid-19"
    else:
        result="Normal"
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        preds = model_predict(file_path, model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

