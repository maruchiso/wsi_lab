import numpy as np
import torch
import torch.nn as nn
import autograd.numpy as anp
from autograd import grad

# Define the function to approximate
def target_function(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

# Generate data points
np.random.seed(42)
x_data = np.linspace(-10, 10, 500).reshape(-1, 1)
y_data = target_function(x_data)

# Split into train and test data
train_size = int(0.8 * len(x_data))
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Build a Multilayer Perceptron (MLP) using PyTorch
class MLP(nn.Module):
    def __init__(self, num_neurons):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, num_neurons)
        self.hidden2 = nn.Linear(num_neurons, num_neurons)
        self.output = nn.Linear(num_neurons, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient-based training using autograd solver
def train_with_gradient(num_neurons):
    model = MLP(num_neurons)
    # Define autograd-compatible function
    def loss_function(weights):
        start = 0
        for param in model.parameters():
            size = param.numel()
            param.data = torch.tensor(weights[start:start+size].reshape(param.shape), dtype=torch.float32)
            start += size
        y_pred = model(x_train_tensor).detach().numpy()
        return mean_squared_error(y_train, y_pred)

    # Gradient descent solver
    def solver(f, x0, alfa=1e-3, epsilon=1e-6, iterations=3000):
        gradient = grad(f)
        x = np.array(x0, dtype=anp.float64)
        for i in range(iterations):
            gradient_value = gradient(x)
            x_new = x - alfa * gradient_value
            if np.all(np.abs(gradient_value) <= epsilon) and np.all(np.abs(x_new - x) <= epsilon):
                break
            x = x_new
        return x

    total_params = sum(p.numel() for p in model.parameters())
    initial_weights = np.random.randn(total_params)
    optimal_weights = solver(loss_function, initial_weights)

    # Set optimal weights in the model
    start = 0
    for param in model.parameters():
        size = param.numel()
        param.data = torch.tensor(optimal_weights[start:start+size].reshape(param.shape), dtype=torch.float32)
        start += size

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).numpy()
        mse = mean_squared_error(y_test, y_pred)
    return mse, model

# Evolutionary optimization of weights using custom solver
def evolutionary_training(num_neurons):
    model = MLP(num_neurons)

    # Define the loss function
    def loss_function(weights):
        start = 0
        for param in model.parameters():
            size = param.numel()
            param.data = torch.tensor(weights[start:start+size].reshape(param.shape), dtype=torch.float32)
            start += size
        y_pred = model(x_train_tensor).detach().numpy()
        return mean_squared_error(y_train, y_pred)

    # Custom evolutionary solver
    def solver(f, x0, max_iteration=30000, a=5, sigma=5, max_no_improve_counter=1000):
        iteration = 1
        success_counter = 0
        function_value = f(x0)
        x = np.array(x0, dtype=anp.float64)
        no_improve_counter = 0

        while iteration <= max_iteration:
            mutation = x + sigma * np.random.normal(0, 1, size=x.shape)
            new_function_value = f(mutation)

            if new_function_value <= function_value:
                success_counter += 1
                function_value = new_function_value
                x = mutation
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= max_no_improve_counter:
                break

            if iteration % a == 0:
                if success_counter / a > 0.2:
                    sigma *= 1.22
                if success_counter / a < 0.2:
                    sigma *= 0.82
                success_counter = 0
            iteration += 1

        return x

    # Initialize weights and run solver
    total_params = sum(p.numel() for p in model.parameters())
    initial_weights = np.random.randn(total_params)
    optimal_weights = solver(loss_function, initial_weights)

    # Set optimal weights in the model
    start = 0
    for param in model.parameters():
        size = param.numel()
        param.data = torch.tensor(optimal_weights[start:start+size].reshape(param.shape), dtype=torch.float32)
        start += size

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).numpy()
        mse = mean_squared_error(y_test, y_pred)
    return mse, model

# Run experiments
num_neurons = [5, 10, 20, 50]
results = []

for neurons in num_neurons:
    mse_gradient, model_gradient = train_with_gradient(neurons)
    mse_evolutionary, model_evolutionary = evolutionary_training(neurons)
    results.append((neurons, mse_gradient, mse_evolutionary))

# Display results
import pandas as pd
results_df = pd.DataFrame(results, columns=['Neurons', 'Gradient MSE', 'Evolutionary MSE'])
import ace_tools as tools; tools.display_dataframe_to_user(name="Comparison of Approximation Methods", dataframe=results_df)
