Objective - WAP to implement a three-layer neural network using Tensor flow library (only, no keras) to classify MNIST handwritten digits dataset. Demonstrate the implementation of feed-forward and back-propagation approaches.

Explanation - This code implements a three-layer **neural network** using **TensorFlow** (without Keras) to classify MNIST handwritten digits. It defines an **input layer (784 neurons), two hidden layers (128 and 64 neurons, ReLU activation), and an output layer (10 neurons)**. The weights and biases are initialized randomly. The **feed-forward** step computes predictions, while **backpropagation** (using Adam optimizer) minimizes the cross-entropy loss. The network is trained over **10 epochs** using mini-batches of **100 images**. After training, the model's accuracy is evaluated on the test dataset. The expected output is the classification accuracy of the model on unseen MNIST test images. 🚀


