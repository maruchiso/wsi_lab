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
    def __init__(self, learning_rate, iterations, lambda_param, kernel='linear', gamma=1, degree=3, coef0=1, x_train = None, y_train = None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.w = None
        self.b = None
        self.kernel = kernel
        self.gamma = gamma #only RBF
        self.degree = degree #only polynomial
        self.coef0 = coef0 #only polynomial
        self.alfa = None
        self.x_train = x_train
        self.y_train = y_train

    
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
        self.alfa = np.zeros(samples)

        for iteration in range(self.iterations):
            
            for sample in range(samples):
                sum_kernel = 0
                for j in range(samples):
                    
                    kernel_value = self.kernel_function(x[j], x[sample])
                    contribution = self.alfa[j] * y[j] * kernel_value

                    sum_kernel += contribution
                
                if y[sample] * (sum_kernel + self.b) < 1:
                    self.alfa += self.learning_rate
                    self.b += self.learning_rate * y[sample]
    
    def predict(self, x_test):
        y_predicted = []
        for x in x_test:
            sum_kernel = 0

            for i in range(len(self.x_train)):
                kernel_value = self.kernel_function(self.x_train[i], x)
                contribution = self.alfa[i] * self.y_train[i] * kernel_value
                sum_kernel += contribution
            
            decision_value = sum_kernel + self.b
            predicted_label = np.sign(decision_value)
            y_predicted.append(predicted_label)
        
        return np.array(y_predicted)





df = pd.read_csv("C:\\Users\\marcin\\Desktop\\wsi_lab\\lab4\\data\\winequality-red.csv", sep=";")
df['quality_binary'] = np.where(df["quality"] >= 6, 1, -1)

x = df.drop(columns=["quality", "quality_binary"])
y = df["quality_binary"].astype(float)
x = x.to_numpy()
y = y.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#my implementation

kernels = ['linear', 'rbf']
for kernel in kernels:
    print(f"Kernel: {kernel}")
    svm = SVM(learning_rate=0.001, iterations=10, lambda_param=0.01, kernel=kernel, gamma=0.1, degree=3, coef0=1, x_train=x_train, y_train=y_train)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for SVM with kernel {kernel}: {accuracy * 100:.2f}%")   
'''
#sklearn implementation
from sklearn.svm import SVC
kernels = ['linear','rbf']

for kernel in kernels:
    print(f"Kernel: {kernel}")
    
    # Implementacja scikit-learn
    if kernel == 'polynomial':
        # W scikit-learn kernel 'poly' odpowiada wielomianowemu
        sklearn_kernel = 'poly'
    else:
        sklearn_kernel = kernel  # 'linear' i 'rbf' są takie same
    
    model = SVC(kernel=sklearn_kernel, gamma=0.1, degree=3, coef0=1, C=1.0)
    model.fit(x_train, y_train)
    sklearn_y_pred = model.predict(x_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_y_pred)
    print(f"Accuracy for sklearn SVM with kernel {kernel}: {sklearn_accuracy * 100:.2f}%")
'''