import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def function(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


#random data
np.random.seed(42)
x_data = np.linspace(-10, 10, 500).reshape(-1, 1)
y_data = function(x_data)

#split data
train_size = int(0.8 * len(x_data))
x_train, x_test = x_data[:train_size], x_data = x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

#numpy arrays to Pytorch arrays
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#MultiLayer Perceptron
class MLP(nn.Module):
    def __init__(self, number_neurons):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(1, number_neurons)
        self.hidden_layer2 = nn.Linear(number_neurons, number_neurons)
        self.output_layer = nn.Linear(number_neurons, 1)
        
    def forward(self, x):
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        
        return x

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#metoda gradientowa
def gradient_training(number_neurons, epochs):
    model = MLP(number_neurons=number_neurons)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = nn.MSELoss(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).numpy()
        mse = mean_squared_error(y_test, y_pred)
    return mse, model

#metoda ewolucyjna

