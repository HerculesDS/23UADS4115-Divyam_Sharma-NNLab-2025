# 1. WAP to implement the Perceptron Learning Algorithm using numpy in Python. Evaluate performance of a single perceptron for NAND and XOR truth tables as input dataset.

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error


nand_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_labels = np.array([1, 1, 1, 0]) 

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([0, 1, 1, 0])



print("Training Perceptron for NAND gate...")
nand_perceptron = Perceptron(input_size=2)
nand_perceptron.train(nand_inputs, nand_labels)

print("Testing Perceptron on NAND gate...")
for inputs in nand_inputs:
    print(f"Input: {inputs}, Predicted: {nand_perceptron.predict(inputs)}")

print("\nTraining Perceptron for XOR gate...")
xor_perceptron = Perceptron(input_size=2)
xor_perceptron.train(xor_inputs, xor_labels)


print("Testing Perceptron on XOR gate...")
for inputs in xor_inputs:
    print(f"Input: {inputs}, Predicted: {xor_perceptron.predict(inputs)}")

'''
Explanation:
NAND Gate:
Linearly Separable: The perceptron can correctly classify inputs for a NAND gate because the problem is linearly separable. In simple terms, you can draw a straight line to separate the input values into their correct categories.

XOR Gate:
Non-Linearly Separable: The perceptron fails to classify an XOR gate correctly because the XOR function is not linearly separable. This means you can't draw a straight line to separate the input values into their correct categories.

Limitations:
Linearly Separable Problems: The perceptron algorithm works well for problems like NAND gates, which are linearly separable.

Non-Linearly Separable Problems: For problems like XOR gates, which are not linearly separable, a single perceptron is not enough. You need more complex networks, such as multi-layer neural networks (often referred to as multi-layer perceptrons or MLPs), to solve these non-linear problems.
'''
