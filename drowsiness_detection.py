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





train_datagen = ImageDataGenerator(validation_split=0.2,
                                   preprocessing_function=preprocess_input) # don't use rescale = 1./255

train_generator = train_datagen.flow_from_directory( data_dir,
                                                     target_size=(224,224),
                                                     batch_size=100,
                                                     shuffle=True,
                                                     class_mode='categorical',
                                                     subset='training')

validation_datagen = ImageDataGenerator(validation_split=0.2,
                                        preprocessing_function=preprocess_input)

validation_generator =  validation_datagen.flow_from_directory( data_dir,
                                                                target_size=(224,224),
                                                                batch_size=100,
                                                                class_mode='categorical',
                                                                subset='validation')












model_res50 = Sequential()

model_res50.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet'
    ))

model_res50.add(Dense(2, activation='softmax'))

model_res50.layers[0].trainable = False 

model_res50.summary()


steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)





model_res50.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model_res50.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    validation_steps=steps_per_epoch_validation,
    epochs=7,
    validation_data=validation_generator,
    verbose=1
)


model_res50.save('drowsines_model_1_ResNet50_Binary2.h5')







img_array = cv2.imread('../input/drowsiness-detection/open_eye/s0001_01844_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
new_array = cv2.resize(backtorgb, (224, 224))
X_input = np.array(new_array).reshape(1, 224, 224, 3)
# plt.imshow(new_array)
prediction = model_res50.predict(X_input/255.0)
prediction





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




