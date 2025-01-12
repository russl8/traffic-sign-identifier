import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    print(images[0])
    # Split data into training and testing sets
    # labels = tf.keras.utils.to_categorical(labels)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     np.array(images), np.array(labels), test_size=TEST_SIZE
    # )
    # a = tf
    # a.models
    # # Get a compiled neural network
    # model = get_model()

    # # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS)

    # # Evaluate neural network performance
    # model.evaluate(x_test,  y_test, verbose=2)

    # # Save model to file
    # if len(sys.argv) == 3:
    #     filename = sys.argv[2]
    #     model.save(filename)
    #     print(f"Model saved to {filename}.")


def load_data(data_dir: str) -> (
        tuple[
            list[np.ndarray], list[int]
        ]):
    """
    returns a list of resized images of type ndarray, and another list of corresponding labels.
    """
    images: list[np.ndarray] = [] # a list of .ppm images converted into ndarrays.
    labels: list[int] = []  # the sign corresponding to an image with each sign being a number [0-42]

    for sign_category_number in range(NUM_CATEGORIES):
        sign_dirpath = os.path.join(data_dir, str(sign_category_number))
        if os.path.isdir(sign_dirpath):
            for image_filename in os.listdir(sign_dirpath):
                image_filepath = os.path.join(sign_dirpath, image_filename)
                image: np.ndarray = cv2.imread(image_filepath) # read image using cv2 and convert into an ndarray
                resized_image = cv2.resize(image,(IMG_WIDTH,IMG_HEIGHT))
                images.append(resized_image)
                labels.append(sign_category_number)
                break
        else:
            raise Exception(f"Directory {sign_dirpath} does not exist.")
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
