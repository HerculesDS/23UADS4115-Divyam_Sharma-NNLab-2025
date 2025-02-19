import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define network architecture
input_size = 784   # 28x28 pixels (flattened)
hidden1_size = 128  # First hidden layer size
hidden2_size = 64   # Second hidden layer size
output_size = 10   # Output layer size (digits 0-9)
learning_rate = 0.01  # Learning rate for optimizer

# Define placeholders for inputs and outputs
X = tf.placeholder(tf.float32, [None, input_size])  # Input images
y = tf.placeholder(tf.float32, [None, output_size])  # Labels (one-hot encoded)

# Initialize weights and biases
weights = {
    'W1': tf.Variable(tf.random_normal([input_size, hidden1_size])),  # Weights from input to hidden layer 1
    'W2': tf.Variable(tf.random_normal([hidden1_size, hidden2_size])),  # Weights from hidden layer 1 to hidden layer 2
    'W3': tf.Variable(tf.random_normal([hidden2_size, output_size]))  # Weights from hidden layer 2 to output layer
}
biases = {
    'b1': tf.Variable(tf.zeros([hidden1_size])),  # Bias for hidden layer 1
    'b2': tf.Variable(tf.zeros([hidden2_size])),  # Bias for hidden layer 2
    'b3': tf.Variable(tf.zeros([output_size]))  # Bias for output layer
}

# Feed-forward propagation
def neural_network(X):
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, weights['W1']), biases['b1']))  # First hidden layer with ReLU activation
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['W2']), biases['b2']))  # Second hidden layer with ReLU activation
    output_layer = tf.add(tf.matmul(layer2, weights['W3']), biases['b3'])  # Output layer (logits)
    return output_layer

# Compute logits and loss
logits = neural_network(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))  # Cross-entropy loss

# Backpropagation (Gradient Descent Optimizer)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # Optimizer for minimizing loss

# Accuracy calculation
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))  # Check if predictions match labels
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate accuracy

# Training the neural network
num_epochs = 10  # Number of training epochs
batch_size = 100  # Size of each training batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize all variables
    
    for epoch in range(num_epochs):
        num_batches = mnist.train.num_examples // batch_size  # Determine number of batches per epoch
        for i in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)  # Fetch mini-batch
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})  # Perform optimization
        
        # Calculate loss and accuracy for current epoch
        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X: mnist.train.images, y: mnist.train.labels})
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # Test model on test dataset
    test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
    print(f"Test Accuracy: {test_acc:.4f}")
