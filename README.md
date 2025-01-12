**Traffic Sign Itentifier**

A Convolutional Neural Network (CNN) that classifies road signs, using the the German Traffic Sign Recognition Benchmark (GTSRB) dataset (http://benchmark.ini.rub.de/?section=gtsrb&subsection=news),
containing ~30000 images of 43 different road signs, using OpenCV and TensorFlow.

**Process**

1. Create testing and training data by transforming each image into a structured format using OpenCV to convert each .ppm image to a NumPy array and resizing it, while labelling the image at the same time.
2. Use scikit-learn to split the data into training and testing splits, leaving 40% of the data for testing and shuffling to ensure that the data is randomly distributed.
3. Use one-hot encoding to create a binary vector for each sign category, as there are 43 possible categories (numbered 0-42), to support multi-class classification.
4. Using a Keras Sequential Model as it is good for standard multi-class classification, create the following layers:
  - Conv2D() as the CNN is dealing with a 2d input with the following parameters:
    -  filters=3 , as nothing complex is needed for a smaller dataset
    -  kernel_size=(2,2), shying away from the standard (3,3) as more precision seemed to work better
    - activation=relu to learn more complex patterns within the data
    - input_shape=(30, 30, 3)
    - use_bias=False as batch normalization is being used (will account for some bias there.)
  - MaxPooling2D(pool_size=(2, 2)) to add a little generalization to the images with
  - Flatten() to reshape outputs feed the dense layers
  - Dense() with the following parameters:
    - units=64 as not too many layers are needed for smaller datasets
    - relu activaton
  - Dropout(0.2) as we don't want to remove too many neurons (dataset is not that large)
  - BatchNormalization() to minimize overfitting due to low dropout
  - Final Dense layer with following parameters:
    - units=NUM_CATEGORIES to output a neuron for each category
    - activation=softmax as it is well suited for multiclass classification
5. Compiling the model with the following parameters:
 - optimizer=adam
 - loss=categorical_crossentropy for multiclass classification
 - metrics=["accuracy"]

**Usage**

1. Clone the repositiory
2. Install the GTSRB dataset
3. Setup a virtual environment and install required libraries.
4. Run the program via command line as follows: python traffic.py <GTSRB_directory> <yourmodelname.h5>

**Training & Testing Logs**

![image](https://github.com/user-attachments/assets/ae7d8634-c55c-448f-adda-54caefb9bad9)

**Future Plans**

- Find a way to get to the same accuracy or higher, using less epochs.
- There seems to be a little overfitting present as the accuracy of the testing data < training data, so find a way to reduce that.

