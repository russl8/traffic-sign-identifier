import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 20
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

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, shuffle=True
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    print("Model evaluation using testing dataset: ")
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir: str) -> (
        tuple[
            list[np.ndarray], list[int]
        ]):
    """
    returns a list of resized images of type ndarray, and another list of corresponding labels.
    """
    images: list[np.ndarray] = [
    ]  # a list of .ppm images converted into ndarrays.
    # the sign corresponding to an image with each sign being a number [0-42]
    labels: list[int] = []

    for sign_category_number in range(NUM_CATEGORIES):
        sign_dirpath = os.path.join(data_dir, str(sign_category_number))
        if os.path.isdir(sign_dirpath):
            for image_filename in os.listdir(sign_dirpath):
                image_filepath = os.path.join(sign_dirpath, image_filename)
                # read image using cv2 and convert into an ndarray
                image: np.ndarray = cv2.imread(image_filepath)
                resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(resized_image)
                labels.append(sign_category_number)
        else:
            raise Exception(f"Directory {sign_dirpath} does not exist.")
    return images, labels


def get_model():
    """
    Returns a CNN model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            3,  # convolution layer, use 32 diff. filters on the image where
            (2, 2),  # each filter is a 3x3 kernel
            activation="relu",  # to learn more complex patterns
            # each image is a 30x30 pixel of 3 colours (red,yellow,green)
            input_shape=(30, 30, 3),
            # dont need as batch normalization is being used (will account for some bias there.)
            use_bias=False,

        ),

        # add a little bit of generalization (many images will have slight differences)
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),  # flatten to feed into dense layers
        # dont need too many layers for small dataset
        tf.keras.layers.Dense(64, activation="relu"),
        # dont remove too many neurons as dataset is not that large (15k training elements)
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),  # to reduce overfitting (low dropout)

    
        tf.keras.layers.Dense(
            NUM_CATEGORIES,
            activation="softmax"  # for multiclass classification
        )

    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
