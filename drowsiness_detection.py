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





%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt





from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions for the validation data
y_pred = model_res50.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=validation_generator.class_indices, yticklabels=validation_generator.class_indices)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()






# Plot training and validation accuracy per epoch
plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss per epoch
plt.plot(history.history['loss'], 'r', label='Training Loss')
plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()







from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    """Learning rate scheduler."""
    initial_lr = 0.001
    decay_factor = 0.1
    decay_epochs = 3

    if epoch < decay_epochs:
        return initial_lr
    else:
        return initial_lr * decay_factor

# Create a learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Fit the model with the learning rate scheduler
history = model_res50.fit(train_generator,
                          steps_per_epoch=steps_per_epoch_training,
                          validation_steps=steps_per_epoch_validation,
                          epochs=7,
                          validation_data=validation_generator,
                          verbose=1,
                          callbacks=[lr_scheduler])





# Plot training and validation accuracy on the same plot
plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss on the same plot
plt.plot(history.history['loss'], 'r', label='Training Loss')
plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()









from sklearn.metrics import precision_score, recall_score, f1_score

# Get predictions for the validation data
y_pred = model_res50.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Plot precision, recall, and F1-score
plt.bar(['Precision', 'Recall', 'F1-score'], [precision, recall, f1])
plt.title('Precision, Recall, and F1-score')
plt.show()






from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the positive class
y_pred_prob = model_res50.predict(validation_generator)[:, 1]

# Calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()









# Plot learning curves for accuracy
plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Learning Curves - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot learning curves for loss
plt.plot(history.history['loss'], 'r', label='Training Loss')
plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
plt.title('Learning Curves - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()









from sklearn.metrics import precision_recall_curve

# Get predicted probabilities for the positive class
y_pred_prob = model_res50.predict(validation_generator)[:, 1]

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

# Plot precision-recall curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()




