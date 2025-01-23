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
    def __init__(self, learning_rate, iterations, lambda_param):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.w = None
        self.b = None
        
    

    def fit(self, x, y):
        samples, feautures = x.shape
        self.w = np.zeros(feautures)
        self.b = 0

        for n in range(self.iterations):
            for idx, x_i in enumerate(x):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, x):
        approx = np.dot(x, self.w) - self.b
        return np.sign(approx)





df = pd.read_csv("C:\\Users\\marcin\\Desktop\\wsi_lab\\lab4\\data\\winequality-red.csv", sep=";")
df['quality_binary'] = np.where(df["quality"] >= 6, 1, -1)

x = df.drop(columns=["quality", "quality_binary"])
y = df["quality_binary"].astype(float)
x = x.to_numpy()
y = y.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
'''
svm = SVM(learning_rate=0.001, iterations=1000, lambda_param=0.01)

svm.fit(x=x_train, y=y_train)
y_pred = svm.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność: {accuracy * 100:.2f}%")

'''
# Eksperyment 1: Wpływ learning_rate
learning_rates = [0.0001, 0.001, 0.1]
for lr in learning_rates:
    print(f"Testing learning_rate: {lr}")
    svm = SVM(learning_rate=lr, iterations=1000, lambda_param=0.1)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for learning_rate={lr}: {accuracy * 100:.2f}%")

# Eksperyment 2: Wpływ iterations
iterations = [10, 100, 1000]
for it in iterations:
    print(f"Testing iterations: {it}")
    svm = SVM(learning_rate=0.01, iterations=it, lambda_param=0.1)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for iterations={it}: {accuracy * 100:.2f}%")

# Eksperyment 3: Wpływ lambda_param
lambda_params = [0.01, 0.1, 1]
for lmbd in lambda_params:
    print(f"Testing lambda_param: {lmbd}")
    svm = SVM(learning_rate=0.01, iterations=1000, lambda_param=lmbd)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for lambda_param={lmbd}: {accuracy * 100:.2f}%")

