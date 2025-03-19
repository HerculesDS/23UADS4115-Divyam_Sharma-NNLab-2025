



### Three-Layer Neural Network Performance Evaluation

#### Model Overview
**Training Configurations:**
The model is trained using varied batch sizes (1, 10, 100) and epochs (10, 50, 100) to analyze performance differences.

**Data Preprocessing:**
- **Normalization:** Pixel values are scaled between 0 and 1 for better model convergence.
- **Reshaping:** Each 28x28 image is flattened into a 784-dimensional vector.
- **Label Encoding:** Labels are converted to one-hot encoded vectors to match the output layer's format.

**Model Architecture:**
- **Input Layer:** Contains 784 neurons, representing each pixel in the image.
- **Hidden Layer:** Consists of 256 neurons with Xavier initialization and ReLU activation for efficient learning.
- **Output Layer:** Comprises 10 neurons corresponding to the 10 digit classes, returning raw logits.

**Loss Function:**
The model uses softmax cross-entropy loss, ideal for multi-class classification tasks.

**Optimizer:**
The Adam optimizer with a learning rate of 0.1 is employed for effective weight updates.

**Training Process:**
- The model trains for 50 epochs with a batch size of 10 by default.
- Users can pause training with Ctrl+C and resume later without data loss.
- The average loss for each batch is tracked using a `tqdm` progress bar for clear visibility.

**Performance Analysis:**
- The model's accuracy is evaluated after each epoch.
- The accuracy is compared with the true labels to assess performance.

**Visual Aids:**
- The training progress is illustrated using a loss curve, accuracy curve, and confusion matrix.
- Metrics like total training time and final accuracy are calculated for performance insights.

### Code Description
**Data Loading & Preprocessing:**
- The MNIST dataset is loaded using `tf.keras.datasets.mnist`.
- Pixel values are normalized by dividing by 255.0.
- Images are reshaped into 1D vectors (size 784) and labels are one-hot encoded to ensure compatibility with the output layer.

**Model Definition:**
- A custom `NeuralNetwork` class defines the network structure.
- Layer 1: A fully connected layer with 784 input units and 256 hidden units, using Xavier initialization.
- Layer 2: Another fully connected layer with 256 hidden units and 10 output neurons representing the digit classes.
- ReLU activation is applied in the hidden layer for non-linearity, while raw logits are returned in the output layer.

**Loss & Optimization:**
- The `softmax_cross_entropy_with_logits` function calculates the cross-entropy loss between true labels and logits.
- The Adam optimizer with a learning rate of 0.1 ensures efficient gradient descent updates.

**Training Process:**
- The `train_step` function performs forward propagation, computes the loss, calculates gradients using `tf.GradientTape`, and updates the model's weights.

**Evaluation Process:**
- The `evaluate` function computes model accuracy by passing logits through softmax activation and comparing predictions with true labels.

**Training Pause Feature:**
- A signal handler for `SIGINT` allows the training to pause (via Ctrl+C) and resume when the user presses Enter.

**Training Loop:**
- The loop iterates for the specified number of epochs (default: 50).
- Each epoch divides the data into batches (default size: 10).
- After each batch, the model computes the average loss and evaluates accuracy.
- A progress bar using `tqdm` shows the training status.

**Performance Tracking:**
- Training duration is recorded to assess computational efficiency.

**Visualization Tools:**
- A loss curve visualizes how the model's error decreases during training.
- An accuracy curve shows the model's performance improvements over epochs.

**Confusion Matrix:**
- The final model predictions are compared with true labels to generate a confusion matrix.
- This is presented as a heatmap for visual clarity in identifying misclassifications.

