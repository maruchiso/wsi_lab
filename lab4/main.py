'''
Zaimplementuj algorytm SVM oraz zbadaj działanie algorytmu w zastosowaniu do zbioru danych Wine Quality Data Set.

W celu dostosowania zbioru danych do problemu klasyfikacji binarnej zdyskretyzuj zmienną objaśnianą. Pamiętaj, aby podzielić zbiór danych na zbiór trenujący oraz uczący.

Zbadaj wpływ hiperparametrów na działanie implementowanego algorytmu. W badaniach rozważ dwie różne funkcje jądrowe poznane na wykładzie.

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM():
    def __init__(self, learning_rate, iterations, lambda_param, kernel='linear', gamma=1, degree=3, coef0=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.w = None
        self.b = None
        self.kernel = kernel
        self.gamma = gamma #only RBF
        self.degree = degree #only polynomial
        self.coef0 = coef0 #only polynomial

    
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def rbf_kelner(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
    
    def polynomial_kernel(self, x1, x2):
        return (self.coef0 + np.dot(x1, x2)) ** self.degree
    
    def kernel_function(self, x1, x2):
        if self.kernel == "linear":
            return self.linear_kernel(x1, x2)
        elif self.kernel == "polynomial":
            return self.polynomial_kernel(x1, x2)
        elif self.kernel == "rbf":
            return self.rbf_kelner(x1, x2)
        else:
            raise ValueError("Please write one from those kernels: 'linear', 'polynomial', 'rbf")
        
    
    
    def fit(self, x, y):
        samples, feautures = x.shape
        self.w = np.zeros(feautures)
        self.b = 0

        for n in range(self.iterations):
            for idx, x_i in enumerate(x):
                print(n)
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, x):
        approx = np.dot(x, self.w) - self.b
        return np.sign(approx)

df = pd.read_csv("data/winequality-red.csv", sep=";")
df['quality_binary'] = np.where(df["quality"] >= 6, 1, -1)

x = df.drop(columns=["quality", "quality_binary"])
y = df["quality_binary"].astype(float)
x = x.to_numpy()
y = y.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
svm = SVM(learning_rate=0.001, iterations=1000, lambda_param=0.01)

svm.fit(x=x_train, y=y_train)
y_pred = svm.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność: {accuracy * 100:.2f}%")


