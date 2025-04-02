Description of the Model

Input Layer: 784 neurons (MNIST images are 28x28 pixels, flattened into a vector).
Two Hidden Layers:
Experiments with different configurations of neurons per layer.
Each hidden layer uses ReLU activation.
Output Layer: 10 neurons (for 10 digit classes), using softmax for classification.
Optimization: Adam optimizer minimizes the softmax cross-entropy loss.
Training Strategy:
Mini-batch gradient descent with a batch size of 100.
The dataset is shuffled, batched, and prefetched for optimized performance.

Description of the Code

Checks for GPU availability and enables memory growth.
Disables Eager Execution (for better performance in TensorFlow 1.x-style execution).

Data Loading & Preprocessing
Loads the MNIST dataset using tensorflow_datasets.
Preprocessing function:
Reshapes images into a flat vector (784,).
Normalizes pixel values to the range [0,1].
Converts labels to one-hot encoding.

Defining the Neural Network
Randomly initialized weights and biases for each layer.
Forward Propagation:
Input → Hidden Layer 1 → Hidden Layer 2 → Output Layer.
Uses ReLU activation for hidden layers.
Loss Function:
Softmax cross-entropy loss.
Optimization:
Adam optimizer (learning rate varies across runs).
Accuracy Calculation:
Compares predicted labels to actual labels.

Training & Evaluation
The model is trained for 50 epochs.
Iterates over batches of training data.
Evaluates performance on the test set.
Stores results:
Final Loss, Accuracy, Test Accuracy, Confusion Matrix, Execution Time.
Confusion Matrix is computed to analyze misclassifications.

Hyperparameter Experiments
Loops through different configurations:
5 different hidden layer sizes.
3 different learning rates.
Results are stored in a DataFrame (Pandas) for analysis.
