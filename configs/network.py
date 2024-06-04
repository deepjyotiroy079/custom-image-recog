# import the necessary packages
from configs.utils import augmentation
from configs.utils import normalize_layer
import tensorflow as tf

class MobileNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the pretrained mobilenet feature extractor model
        # and set the base model layers to non-trainable
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(height, width, depth),
            include_top=False,
            weights="imagenet",
        )

        base_model.trainable = False

        # define the input to the classification network with
        # image dimensions
        inputs = tf.keras.Input(shape=(height, width, depth))
        
        # apply augmentation to the inputs and normalize the batch
        # images to [-1, 1] as expected by MobileNet
        x = augmentation()(inputs)
        x = normalize_layer()(x)

        # pass the normalized and augmented images to base model,
        # average the 7x7x1280 into a 1280 vector per image,
        # add dropout as a regularizer with dropout rate of 0.2
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # apply dense layer to convert the feature vector into
        # a prediction of classes per image
        outputs = tf.keras.layers.Dense(classes)(x) 
        # build the keras Model by passing input and output of the
        # model and return the model
        model = tf.keras.Model(inputs, outputs)
        return model