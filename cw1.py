############## Polecenie ##############

#Zaimplementuj algorytm gradientu prostego
#Następnie zbadaj zbieżność algorytmu, używając funkcji kwadratowej, F3, F12 z benchmarku CEC2017
#Dziedzina funkcji celu [-100, 100]^n C R^n, n = 10
#Zbadaj wpływ wartości parametru kroku na zbiezność (wykres par (t, q(xt)), gdzie t krok/nr iteracji) metody oraz czas jej działania.
#Rozważ następujące wartości parametru alfa = {1, 10, 100}

#Do wyliczenia gradientu funkcji możesz skorzystać z pakietu autograd Pamiętaj że implementacja solwera powinna być w stanie zoptymalizować każdą funkcje celu 
# oraz posiadać co najmniej 2 warunki stopu.
#Sugerowana sygrantura funkcji def solver(f, x0, ...) -> result


############## Implementacja ##############

from autograd import grad
import numpy as np
from cec2017.functions import f3, f12
import matplotlib.pyplot as plt

#funkcja kwadratowa
def f(x):
    return sum(x ** 2)

def solver(f, x0, alfa=1e-3, epsilon=1e-1):
    gradient = grad(f)
    x = x0
    f_values = []

    while True:

        gradient_value = gradient(x)
        x_new = x - alfa * gradient_value
        f_values.append(f(x_new))

        if np.all(abs(gradient_value)) <= epsilon or np.all(abs(x_new - x)) <= epsilon:
            print("Znaleziono!")
            break

        x = x_new
    return x, f_values

def plot(f_values, alfa):
    plt.plot(f_values)
    plt.title(f"Zbieżność dla alfa={alfa}")
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu")
    plt.grid(True)
    plt.show()

x0 = np.random.uniform(-100.0, 100.0, 10)
x, f_values = solver(f, x0)
print(x)
plot(f_values, 1e-1)

