import numpy as np

# Step Activation Function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron Class
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return step_function(np.dot(self.weights, x))

    def train(self, X, y, learning_rate=0.1, epochs=100):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        for _ in range(epochs):
            for i in range(X.shape[0]):
                y_pred = step_function(np.dot(self.weights, X[i]))
                self.weights += learning_rate * (y[i] - y_pred) * X[i]

# XOR Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR output

# Train Hidden Layer Perceptrons Separately
hidden_perceptron1 = Perceptron(input_size=2)
hidden_perceptron2 = Perceptron(input_size=2)

hidden_y1 = np.array([0, 1, 1, 1])  # NAND-like function
hidden_y2 = np.array([1, 1, 1, 0])  # OR-like function

hidden_perceptron1.train(X, hidden_y1)  # Train independently
hidden_perceptron2.train(X, hidden_y2)  # Train independently

# Get outputs of the hidden layer after training
hidden_outputs = np.array([
    [hidden_perceptron1.predict(x), hidden_perceptron2.predict(x)] for x in X
])

# Train Output Layer Perceptron Separately
output_perceptron = Perceptron(input_size=2)
output_perceptron.train(hidden_outputs, y)  # Train separately on hidden outputs

# Test XOR Function
final_predictions = [output_perceptron.predict([hidden_perceptron1.predict(x), hidden_perceptron2.predict(x)]) for x in X]

# Print Results
print(f"XOR Predictions: {final_predictions}")  # Expected: [0, 1, 1, 0]
