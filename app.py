import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from configs.utils import prepare_batch_dataset
from configs.utils import clean_uploads_folder
from configs import config
from configs.network import MobileNet
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.static_folder = 'static'
dir_path = os.path.dirname(os.path.realpath(__file__))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = tf.keras.utils.load_img(
                filepath, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
            )

            model = tf.keras.models.load_model(config.TRAINED_MODEL_PATH)

            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])


            train_ds_wo_shuffle = prepare_batch_dataset(
                    config.TRAIN_DATA_PATH,
                    config.IMAGE_SIZE,
                    config.BATCH_SIZE,
                    shuffle=False
                )

            # loss, accuracy = model.evaluate(train_ds_wo_shuffle)
            # print("Test accuracy :", accuracy)
            # fetch class names
            class_names = train_ds_wo_shuffle.class_names

            # print(class_names)    
            vegetable_name = class_names[np.argmax(score)]
            vegetable_name_accuracy = 100 * np.max(score)
            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(vegetable_name, vegetable_name_accuracy)
            )
            clean_uploads_folder(filepath=filepath)
            return render_template('index.html', vegetable_name=vegetable_name, vegetable_name_accuracy=vegetable_name_accuracy)
    
    return render_template('index.html')
    
if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
