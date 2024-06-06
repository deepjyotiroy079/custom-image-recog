# USAGE
# python test.py
# import the necessary packages
from configs.utils import prepare_batch_dataset
from configs.utils import callbacks
from configs import config
from configs.network import MobileNet
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
"""
    # build the test dataset pipeline with and without shuffling the dataset
    print("[INFO] building the test dataset with and without shuffle...")
    test_ds_wo_shuffle = prepare_batch_dataset(
        config.TEST_DATA_PATH,
        config.IMAGE_SIZE,
        config.BATCH_SIZE,
        shuffle=False
    )
    test_ds_shuffle = prepare_batch_dataset(
        config.TEST_DATA_PATH, config.IMAGE_SIZE, config.BATCH_SIZE
    )

    # load the trained image classification convolutional neural network
    model = tf.keras.models.load_model(config.TRAINED_MODEL_PATH)
    # print model summary on the terminal
    print(model.summary())
    # evaluate the model on test dataset and print the test accuracy
    loss, accuracy = model.evaluate(test_ds_wo_shuffle)
    print("Test accuracy :", accuracy)
    # fetch class names
    class_names = test_ds_wo_shuffle.class_names
    print('Class Names : ', class_names)
    # generate classification report by (i) predicting on test dataset
    # (ii) take softmax of predictions (iii) for each sample in test set
    # fetch index with max. probability (iv) create a vector of true labels
    # (v) pass ground truth, prediction for test data along with class names
    # to classification_report method
    test_pred = model.predict(test_ds_wo_shuffle)
    test_pred = tf.nn.softmax(test_pred)
    test_pred = tf.argmax(test_pred, axis=1)
    test_true_labels = tf.concat(
        [label for _, label in test_ds_wo_shuffle], axis=0
    )
    print(
        classification_report(
            test_true_labels, test_pred, target_names=class_names
        )
    )


    # Retrieve a batch of images from the test set and run inference
    image_batch, label_batch = test_ds_shuffle.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch)
    # Apply a softmax
    score = tf.nn.softmax(predictions)
    print(score.shape)
    # save the plot for model prediction along with respective test images
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[np.argmax(score[i])])
        plt.axis("off")
    plt.savefig(config.TEST_PREDICTION_OUTPUT) 
"""
img = tf.keras.utils.load_img(
    'uploads\\0001.jpg', target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
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