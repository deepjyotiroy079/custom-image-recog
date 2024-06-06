from configs import config
import tensorflow as tf
import os

"""
    It then uses tf.keras.preprocessing.image_dataset_from_directory to load 
    the images from the directory, resizes them to the given image size, shuffles 
    them if shuffle is True, and returns the dataset in the form of batches.
"""
def prepare_batch_dataset(data_path, img_size, batch_size, shuffle=True):
	return tf.keras.preprocessing.image_dataset_from_directory(
		data_path,
		image_size=(img_size, img_size),
		shuffle=shuffle,
		batch_size=batch_size
	)
 
 

"""
    The callback monitors the validation loss and stops the training if it does not improve after two epochs.
""" 
def callbacks():
	# build an early stopping callback and return it
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_loss",
			min_delta=0,
			patience=2,
			mode="auto",
		),
	]
	return callbacks

"""
    the function normalize_layer returns a normalization layer. 
    It normalizes the input data by dividing by 127.5 and subtracting 1. 
    In short, it normalizes or scales the input images from [0, 255] to [-1, 1], 
which is what the MobileNetV2 network expects.
"""
def normalize_layer(factor=1./127.5):
	# return a normalization layer
	return tf.keras.layers.Rescaling(factor, offset=-1)


"""
    the function augmentation creates a sequential model with image augmentations: random horizontal flipping, 
    rotation, and zoom. The augmentation helps avoid overfitting and allows the model to generalize better.
"""
def augmentation():
	# build a sequential model with augmentations
	data_aug = tf.keras.Sequential(
		[
			tf.keras.layers.RandomFlip("horizontal"),
			tf.keras.layers.RandomRotation(0.1),
			tf.keras.layers.RandomZoom(0.1),
		]
	)
	return data_aug

"""
	cleaning up the uploads folder
"""
def clean_uploads_folder(filepath):
    os.remove(filepath)