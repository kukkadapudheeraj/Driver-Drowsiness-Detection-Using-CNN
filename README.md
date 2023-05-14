## Driver Drowsiness Detection

This is a drowsiness detection model built with CNN and trained on a dataset of images that contain open and closed eyes.

### Model building process:
The driver drowsiness detection model is build using the following steps:
1) Preprocess the data using ImageDataGenerator for training and validation data.
2) Create a deep learning model architecture (in this case, ResNet50).
3) Compile the model by specifying the optimizer, loss function, and evaluation metrics.
4) Train the model using the fit() method with the training and validation data generators.
5) Evaluate the model's performance using the validation data.

![model building process](https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/961b4e7b-de20-4b9b-9723-a5a1110f0f56)

### Data set description and Access to the dataset

The dataset contains 2 directories **closed_eye and open_eye**, which correspond to the two classes of images that this model distinguishes between.

**steps to access the dataset:**
1) Download the dataset from the following kaggle link: ''
2) Place the dataset in a directory
3) update the following variable in the code with your local path for the directory:

```data_dir = '/input/drowsiness-detection'```

**Data Processing**
The data is pre-processed into size of 224X224 in the batches of 100 for both training and validation purposes using the following code:
```
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

```

**Model Architecture**

**The model architecture is as follows:**

1) A ResNet50 layer with weights pre-trained on ImageNet is added to the model as the base layer, with the include_top parameter set to False to remove the last layer of the model.
2) A Dense layer with 2 output nodes and a softmax activation function is added to the model to output class probabilities.
3) The ResNet50 layer is set to be non-trainable.
4) The model is compiled with an Adam optimizer, categorical cross-entropy loss, and accuracy metric.

## Training
The model is trained using an instance of ImageDataGenerator from the `tensorflow.keras.preprocessing.image` module to generate batches of augmented and preprocessed images from the specified directory. A separate generator with same specifications is created for validation data.

The model is trained for 7 epochs with a validation split of 20%, a batch size of 100, and a target image size of 224x224.

`history = model_res50.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    validation_steps=steps_per_epoch_validation,
    epochs=7,
    validation_data=validation_generator,
    verbose=1
)`
### The Flow Chart of our deep learning model is as follows:

![algorithem_psudocode](https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/cf57c323-2edd-4e02-a13b-a14413abddd6)
## Results

***After training the model for 7 times the model has acheived an accuracy of 0.9949 and loss of 0.0155.***

<img width="500" alt="Learning Curves-Loss" src="https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/8c4709f3-117f-456a-a4fd-fb4a13777cd3"><br>
![training_and_validation-accuracy](https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/83891042-81df-4687-8bec-6cbe1609d485)

![training_and_validation-loss](https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/a74d936e-2877-4740-bd98-a53a9dfd6c25)
<br>
<img width="500" alt="ROC_curve" src="https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/256e3113-0c76-4ece-8342-b835d8031707"><br>
<img width="500" alt="precision_recall_f1" src="https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/ac521391-12ea-4cca-893c-32c0d34e17dc"><br>
<img width="500" alt="confusion_matrix" src="https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/b4fe0334-66c5-473d-8469-5a390e9e9325"><br>
<img width="500" alt="precision_recall_curve" src="https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/f51d47e9-f95e-4a45-9a55-252b986a2f91"><br>
<img width="500" alt="Learning Curves-Accuracy" src="https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/9341968a-ab76-4d77-81fa-ace31fee3d9d">





