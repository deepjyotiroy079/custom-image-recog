from tensorflow.python.keras.backend import set_session
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, send_file, url_for
from keras.models import load_model
import os
from PIL import Image, ImageOps
import numpy as np
import math
from configs import config

app = Flask(__name__)
app.static_folder = 'static'
dir_path = os.path.dirname(os.path.realpath(__file__))


MODEL_PATH = os.path.join(os.getcwd(), 'output\\vegetable_classifier.keras')
model = load_model(MODEL_PATH)


sess = tf.compat.v1.Session()

graph = tf.compat.v1.get_default_graph()
set_session(sess)


# print(MODEL_PATH)
print(model.summary())

@app.route('/', methods=['GET', 'POST'])
def index():
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        np.set_printoptions(suppress=True)
        
        if request.method == 'POST':
            print("POST to /")
            file = request.files['query_img']

            img = Image.open(file.stream)  # PIL image
            # uploaded_img_path = "static\\uploads\\" + file.filename
            uploaded_img_path = os.path.join(os.getcwd(), 'uploads', file.filename)
            # uploaded_img_path = os.path.join(os.getcwd(), 'uploads', file.filename)
            img.save(uploaded_img_path)
            
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            # Replace this with the path to your image
            # uploaded_img_path = os.path.join(os.getcwd(), 'uploads', 'IM-0033-0001-0001.jpg')

            print("Image path {}".format(uploaded_img_path))
            image = Image.open(uploaded_img_path)

            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
            image = ImageOps.fit(image, size)
            image = image.convert('RGB')
            # image.show()
            #turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)

            # print("Prediction....")
            # print(prediction)

            print("Positive : {}".format(float(prediction[0][0])))
            # positive_score = float(prediction[0][0])
            print("Negative : {}".format(float(prediction[0][1])))
            # negative_score = float(prediction[0][1])

            predicted_scores = [{
                "positive": round(float(math.floor(prediction[0][0]*100)), 2),
                "negative": round(float(math.floor(prediction[0][1]*100)), 2)
            }]
            return render_template('index.html', query_path=uploaded_img_path, predicted_scores=predicted_scores)
            
        else:
            return render_template('index.html')


if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)