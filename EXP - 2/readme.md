Objective - WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.

Explanation - This code implements a simple neural network to solve the XOR problem using perceptrons. A single perceptron can't solve XOR since it's not linearly separable, so the code creates a hidden layer with two perceptrons. Their outputs are then fed into a third perceptron (output layer), which learns to combine them to correctly compute XOR. The perceptrons are trained using a basic weight update rule. After training, the network correctly predicts XOR values: [0, 1, 1, 0]. This approach manually builds a multi-layer perceptron (MLP).


Limitation - 

	Step Function Restriction – The perceptrons use a step activation function, which is not differentiable, preventing gradient-based optimization (e.g., backpropagation).

	Manual Feature Engineering – The hidden layer is manually designed to mimic NAND and OR, instead of learning representations automatically.

	Fixed Learning Approach – The perceptron learning rule only works for linearly separable functions at each layer, limiting scalability.
