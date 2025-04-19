<h1>Object 5</h1> <h4>Write a program to train and evaluate a Convolutional Neural Network using the Keras Library to classify the Fashion MNIST dataset. Demonstrate the effects of filter size, regularization, batch size, and optimization algorithms on model performance.</h4> <hr> <h2>Model Description</h2>
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset.

Dataset
Fashion MNIST: 28x28 grayscale images categorized into 10 fashion classes.

Preprocessing Steps:

Pixel values normalized to the [0, 1] range.

Input reshaped to (28, 28, 1) to suit CNN input requirements.

Labels one-hot encoded.

Data Augmentation applied using ImageDataGenerator:

Includes random rotation, zoom, shift, and shear transformations.

Model Architecture
Input Layer: (28, 28, 1)

Convolution Block 1:

Two Conv2D layers → BatchNormalization → ReLU

MaxPooling2D

Dropout

Convolution Block 2 (Residual Block):

Two Conv2D layers → BatchNormalization

Residual connection using Add() followed by ReLU

MaxPooling2D

Dropout

Convolution Block 3:

Conv2D

GlobalAveragePooling2D

Fully Connected Layers:

Dense(512) → BatchNormalization → ReLU → Dropout

Dense(10) with Softmax activation for classification output

Training Configuration
Optimizer: Adam with Exponential Learning Rate Decay

)
Loss Function: CategoricalCrossentropy with label_smoothing=0.1

Training & Evaluation
The model is trained using augmented data.

Validation accuracy is monitored after each epoch.

The best model (based on validation performance) is saved to 'best_model.h5'.

Final model is evaluated on the test set.

Visualization
Accuracy Curve: Plots training and validation accuracy.

Loss Curve: Plots training and validation loss.

<hr> <h2>My Comments</h2> <ul> <li>The model achieved a maximum test accuracy of **92.53%**.</li><br> <li>The addition of **Batch Normalization** and **Dropout layers**, along with **data augmentation**, significantly improved performance, pushing accuracy close to 93%.</li><br><li> with GPU also it takes a lot of time to train</li> </li> </ul>
