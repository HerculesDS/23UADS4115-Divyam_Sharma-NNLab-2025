Experiment 6: Implementation of RNN using PyTorch Library
1. Objective
To build, train, and evaluate a Recurrent Neural Network (RNN) using PyTorch to predict the next day's temperature using historical daily minimum temperature data from Melbourne.

2. Description of the Model:
The model used in this project is a Recurrent Neural Network (RNN) with the following characteristics:

Input Size: 1 (temperature)
Sequence Length: 30 (days)
Hidden Size: 50
Layers: 1
Output: 1 (next day's predicted temperature)
The RNN learns temporal dependencies from historical data and predicts the next value based on previous 30-day temperature sequences.

3. Description of Code:
The implementation includes the following major steps:

Data Loading and Normalization:

Load daily temperature data using pandas.
Normalize the data using MinMaxScaler.
Sequence Creation:

Create input-output pairs using a sliding window approach with a sequence length of 30.
Train-Test Split:

80% training and 20% testing split.
Data is converted to PyTorch tensors and reshaped for RNN input.
Model Definition:

A custom RNNModel class using PyTorch's nn.RNN and nn.Linear.
Training:

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Accuracy Metric: MAE-based (converted to percentage accuracy)
Trains the model over 200 epochs and stores MSE loss and accuracy per epoch.
Evaluation:



4. Performance Evaluation:
Final Test Accuracy :92.4%
Loss Curve: Shows steady convergence of training MSE.
Accuracy Curve: Reflects consistent performance improvements.
Prediction Plot: Shows predicted vs actual temperature for test data visually aligning well.

6. My Comments
The RNN used is simple and may not capture complex long-term patterns in our case there were many rows hence it was unable to capture the pattern correctly .
Model can be improved with more hyperparameter tuning, dropout regularization, or early stopping. 
Overall, the model performs well for a basic RNN and offers a good baseline for more advanced time series models.

