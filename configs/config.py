import os

# define the base path, paths to separate train, validation, and test splits
BASE_DATASET_PATH = "data/Vegetable Images"
TRAIN_DATA_PATH = os.path.join(BASE_DATASET_PATH, "train")
VALID_DATA_PATH = os.path.join(BASE_DATASET_PATH, "validation")
TEST_DATA_PATH = os.path.join(BASE_DATASET_PATH, "test")
OUTPUT_PATH = "output"

# define the image size and the batch size of the dataset
IMAGE_SIZE = 224
BATCH_SIZE = 32
# number of channels, 1 for gray scale and 3 for color images
CHANNELS = 3
# define the classifier network learning rate
LR_INIT = 0.0001
# number of epochs for training
NUM_EPOCHS = 20
# number of categories/classes in the dataset
N_CLASSES = 15
# define paths to store training plots, testing prediction and trained model
ACCURACY_LOSS_PLOT_PATH = os.path.join("output", "accuracy_loss_plot.png")
TRAINED_MODEL_PATH = os.path.join("output", "vegetable_classifier.keras")
TEST_PREDICTION_OUTPUT = os.path.join("output", "test_prediction_images.png")