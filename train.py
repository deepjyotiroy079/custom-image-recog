# USAGE
# python train.py
# import the necessary packages
from configs.utils import prepare_batch_dataset
from configs.utils import callbacks
from configs import config
from configs.network import MobileNet
from matplotlib import pyplot as plt
import tensorflow as tf
import os
# build the training and validation dataset pipeline
print("[INFO] building the training and validation dataset...")
train_ds = prepare_batch_dataset(
	config.TRAIN_DATA_PATH, config.IMAGE_SIZE, config.BATCH_SIZE
)
val_ds = prepare_batch_dataset(
	config.VALID_DATA_PATH, config.IMAGE_SIZE, config.BATCH_SIZE
)

"""
    a pipeline for the training and validation datasets is built. 
    The function prepare_batch_dataset is used to create the dataset by passing in the path of the training data, 
    image size, and batch size, as specified in the config file. The same is done for the validation dataset.
"""
# build the output path if not already exists
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)
# initialize the callbacks, model, optimizer and loss
print("[INFO] compiling model...")
callbacks = callbacks()
model = MobileNet.build(
	width=config.IMAGE_SIZE,
	height=config.IMAGE_SIZE,
	depth=config.CHANNELS,
	classes=config.N_CLASSES
)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LR_INIT)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# compile the model
model.compile(
	optimizer=optimizer,
	loss=loss,
	metrics=["accuracy"]
)

# evaluate the model initially
(initial_loss, initial_accuracy) = model.evaluate(val_ds)
print("initial loss: {:.2f}".format(initial_loss))
print("initial accuracy: {:.2f}".format(initial_accuracy))
# train the image classification network
print("[INFO] training network...")
history = model.fit(
	train_ds,
	epochs=config.NUM_EPOCHS,
	validation_data=val_ds,
	callbacks=callbacks,
)


# save the model to disk
print("[INFO] serializing network...")
model.save(config.TRAINED_MODEL_PATH)
# save the training loss and accuracy plot
plt.style.use("ggplot")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(config.ACCURACY_LOSS_PLOT_PATH)