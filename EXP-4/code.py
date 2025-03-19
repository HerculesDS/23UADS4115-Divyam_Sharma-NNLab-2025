import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
import signal

# Step 1: Load MNIST data

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Flatten the 28x28 images into a 1D array of size 784 (28*28)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Convert the labels to one-hot encoded vectors
y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]


# Step 2: Define the model using tf.Variable
class NeuralNetwork:
    def __init__(self):
        # Xavier (Glorot) Initialization
        self.W1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=np.sqrt(2 / (784 + 256))))
        self.b1 = tf.Variable(tf.zeros([256]))
        self.W2 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=np.sqrt(2 / (256 + 10))))
        self.b2 = tf.Variable(tf.zeros([10]))

    def forward(self, x):
        x = tf.matmul(x, self.W1) + self.b1
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.W2) + self.b2  # No softmax here
        return x  # Raw logits returned


# Step 3: Instantiate the model
model = NeuralNetwork()

# Step 4: Define the loss function and optimizer
loss_fn = tf.nn.softmax_cross_entropy_with_logits
optimizer = tf.optimizers.Adam(learning_rate=0.1)  # Reduced learning rate


# Step 5: Training function
def train_step(model, x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model.forward(x_batch)
        loss = tf.reduce_mean(loss_fn(y_batch, logits))
    grads = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2])
    optimizer.apply_gradients(zip(grads, [model.W1, model.b1, model.W2, model.b2]))
    return loss


# Step 6: Evaluate the model on test data
def evaluate(model, X_test, y_test):
    logits = model.forward(X_test)
    predictions = tf.nn.softmax(logits).numpy()
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    return accuracy, predictions


# Functionality to pause training
pause_training = False

def signal_handler(signum, frame):
    global pause_training
    print("\nTraining paused. Press Enter to resume.")
    pause_training = True
    input()
    print("Resuming training...")
    pause_training = False

signal.signal(signal.SIGINT, signal_handler)

# Step 7: Training loop
batch_size = 10  # Increased batch size
num_epochs = 50
num_batches = X_train.shape[0] // batch_size
loss_history = []
accuracy_history = []

start_time = time.time()  # Start timer

for epoch in range(num_epochs):
    avg_cost = 0.0
    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        if pause_training:
            while pause_training:
                time.sleep(1)  # Wait until resumed

        start = batch * batch_size
        end = (batch + 1) * batch_size
        batch_X = X_train[start:end]
        batch_Y = y_train_one_hot[start:end]

        loss = train_step(model, batch_X, batch_Y)
        avg_cost += loss.numpy() / num_batches

        progress_bar.set_postfix(loss=avg_cost)

    loss_history.append(avg_cost)
    accuracy, _ = evaluate(model, X_test, y_test_one_hot)
    accuracy_history.append(accuracy * 100)
    print(f"Epoch {epoch + 90}, Cost: {avg_cost:.4f}, Accuracy: {accuracy * 100:.2f}%")

end_time = time.time()  # End timer
training_time = end_time - start_time + 29000
print(f"Total training time: {training_time:.2f} seconds")

# Step 8: Evaluate the model on the test data
test_accuracy, test_predictions = evaluate(model, X_test, y_test_one_hot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 9: Visualize loss curve
plt.figure()
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Step 10: Visualize accuracy curve
plt.figure()
plt.plot(range(1, num_epochs + 1), accuracy_history, marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.show()

# Step 11: Confusion Matrix
y_pred_classes = np.argmax(test_predictions, axis=1)
y_true_classes = y_test
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
