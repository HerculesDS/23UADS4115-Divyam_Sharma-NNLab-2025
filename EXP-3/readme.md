Objective - WAP to implement a three-layer neural network using Tensor flow library (only, no keras) to classify MNIST handwritten digits dataset. Demonstrate the implementation of feed-forward and back-propagation approaches.

Explanation - This code implements a three-layer **neural network** using **TensorFlow** (without Keras) to classify MNIST handwritten digits. It defines an **input layer (784 neurons), two hidden layers (128 and 64 neurons, ReLU activation), and an output layer (10 neurons)**. The weights and biases are initialized randomly. The **feed-forward** step computes predictions, while **backpropagation** (using Adam optimizer) minimizes the cross-entropy loss. The network is trained over **10 epochs** using mini-batches of **100 images**. After training, the model's accuracy is evaluated on the test dataset. The expected output is the classification accuracy of the model on unseen MNIST test images. 

**Code Features**
  Dataset: MNIST handwritten digit dataset.
  Data Preprocessing:
        Normalization (scaling pixel values to [0,1]).
        Flattening images (28×28 → 784).
  Neural Network Architecture:
        Input Layer: 784 neurons.
        Hidden Layer: 128 neurons with sigmoid activation.
        Output Layer: 10 neurons with softmax activation.
  Weight Initialization:
        Random initialization for weights (W1, W2).
        Zero initialization for biases (b1, b2).
          Loss Function: Sparse Categorical Crossentropy.
  Optimizer: Adam Optimizer.
  Training:
          Mini-batch gradient descent (batch size = 32).
          20 epochs.
          Gradient Tape for backpropagation.
  Evaluation: Test accuracy computation using argmax for prediction.
  Visualization: Loss curve plotted using matplotlib.
  **Comment**

  The model may take a lot of time to train for e.g. A 6gb vram graphics card (Nvidia RTX 4050 Laptop GPU) takes around 5 min.
  Instead of using the sigmoid function as activation function, rectified linear unit (ReLu) function can be used which is comparatively simple and hence improves training time.
  More layer can be added which has incresae accuracy and made model more complex
