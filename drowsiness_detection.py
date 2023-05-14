!ls /kaggle/input/drowsiness-detection


import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.image import imread
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import models, layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense



data_dir = '/kaggle/input/drowsiness-detection'





# Define an instance of ImageDataGenerator for training data with a validation split of 20% and preprocess the input images using preprocess_input
train_datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)

# Define a generator for training data by calling flow_from_directory method on train_datagen
# flow_from_directory generates batches of augmented and preprocessed images from the specified directory
train_generator = train_datagen.flow_from_directory(data_dir, target_size=(224,224), batch_size=100, shuffle=True, class_mode='categorical', subset='training')

# Define an instance of ImageDataGenerator for validation data with a validation split of 20% and preprocess the input images using preprocess_input
validation_datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)

# Define a generator for validation data by calling flow_from_directory method on validation_datagen
# flow_from_directory generates batches of preprocessed images from the specified directory
validation_generator = validation_datagen.flow_from_directory(data_dir, target_size=(224,224), batch_size=100, class_mode='categorical', subset='validation')











# Create a Sequential model instance
model_res50 = Sequential()

# Add a ResNet50 layer to the model with some specific configuration
model_res50.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet'
    ))

# Add a Dense layer with 2 output nodes and a softmax activation function
model_res50.add(Dense(2, activation='softmax'))

# Set the ResNet50 layer to be non-trainable
model_res50.layers[0].trainable = False 

# Print a summary of the model's architecture
model_res50.summary()

# Determine the number of steps per epoch for training and validation generators
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)






# Compile the model with an Adam optimizer, categorical cross-entropy loss, and accuracy metric
model_res50.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the fit() method and the training and validation generators.
# Set the number of steps per epoch for training and validation, the number of epochs to train for,
# the validation data to use, and the verbosity level (1 for progress bar).
history = model_res50.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    validation_steps=steps_per_epoch_validation,
    epochs=3,
    validation_data=validation_generator,
    verbose=1
)






# Display plots inline in the Jupyter Notebook
%matplotlib inline

# Import the required libraries
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Retrieve the accuracy, validation accuracy, loss, and validation loss values from the history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a range object representing the number of epochs
epochs = range(len(acc))

# Plot the training and validation accuracy over the number of epochs
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()

# Create a new figure and plot the training and validation loss over the number of epochs
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

# Display the plots
plt.show()










# This line saves the model in 'drowsines_model_1_ResNet50_Binary2.h5' file
model_res50.save('drowsines_model_1_ResNet50_Binary2.h5')







# Load the image file in grayscale using OpenCV
img_array = cv2.imread('../input/drowsiness-detection/open_eye/s0001_01844_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)

# Convert the grayscale image to a 3-channel RGB image
backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

# Resize the image to the required input shape of the model (224x224) using OpenCV
new_array = cv2.resize(backtorgb, (224, 224))

# Convert the image to a 4D numpy array with shape (1, 224, 224, 3) to match the input shape of the model
X_input = np.array(new_array).reshape(1, 224, 224, 3)

# Normalize the input image data by scaling pixel values to be between 0 and 1
X_input = X_input / 255.0

# Use the trained model to predict the class probabilities of the input image
prediction = model_res50.predict(X_input)

# Print the predicted class probabilities
print(prediction)



