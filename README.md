## Drowsiness Detection

This is a drowsiness detection model built with CNN and trained on a dataset of images that contain open and closed eyes.

### Data set description and Access to the dataset

The dataset contains 2 directories **closed_eye and open_eye**, which correspond to the two classes of images that this model distinguishes between.

**steps to access the dataset:**
1) Download the dataset from the following google drive link: 
2) Place the dataset in a directory
3) update the following variable in the code with your local path for the directory:

`data_dir = '/input/drowsiness-detection'`

**Model Architecture**

**The model architecture is as follows:**

1) A ResNet50 layer with weights pre-trained on ImageNet is added to the model as the base layer, with the include_top parameter set to False to remove the last layer of the model.
2) A Dense layer with 2 output nodes and a softmax activation function is added to the model to output class probabilities.
3) The ResNet50 layer is set to be non-trainable.
4) The model is compiled with an Adam optimizer, categorical cross-entropy loss, and accuracy metric.

![image](https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/ab296364-235c-43b7-861d-718a5b28a3d7)

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

## Results
The training and validation accuracy and loss are plotted over the number of epochs using `matplotlib`. The final training accuracy is 99.6% and the final validation accuracy is 98.3%.

![image](https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/93ee8137-961f-4643-868a-d29a9e17ebdf)

![image](https://github.com/kukkadapudheeraj/Driver-Drowsiness-Detection-Using-CNN/assets/39270552/6a96e46a-9439-4662-88cf-113c28fc0c6b)

