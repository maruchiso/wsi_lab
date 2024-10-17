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

#funkcja kwadratowa
def f(x):
    return x ** 2

def solver(f, x0, alfa=1e-2, epsilon=1e-6):
    gradient = grad(f)
    x = x0

    while True:
        gradient_value = gradient(x)
        x_new = x - alfa * gradient_value
        if abs(gradient_value) <= epsilon or abs(x_new - x) <= epsilon:
            break
        x = x_new
        print(x)
    return x
print(solver(f, -0.5))

