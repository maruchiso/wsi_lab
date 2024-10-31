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
import autograd.numpy as np
from cec2017.functions import f3, f12
import matplotlib.pyplot as plt
import time 
import json

#quadratic function
def f(x):
    return np.sum(x ** 2)

def solver(f, x0, alfa=1e-3, epsilon=1e-6, iterations=3000):
    gradient = grad(f)
    x = x0
    f_values = []
    iteration = 0
    for i in range(iterations):
        gradient_value = gradient(x)
        x_new = x - alfa * gradient_value
        #values for the plot
        f_values.append(f(x_new))
        
        #stopping criterion
        if np.all(np.abs(gradient_value) <= epsilon) and np.all(np.abs(x_new - x) <= epsilon):
            print("Znaleziono!")
            break
        
        #update x
        x = x_new
        iteration += 1
        print(i)

    result = {
        'x': x,
        'y': f(x),
        't': iteration,
        'f_values': f_values
    }
    return result

#plot drawing
def plot(f, x0, alfa_to_test):
    evaluation_times = {}
    for alfa in alfa_to_test:
        start_time = time.time()
        result = solver(f, x0, alfa)
        end_time = time.time()
        evaluetion_time = end_time - start_time
        evaluation_times[alfa] = round(evaluetion_time, 3)
        plt.plot(result['f_values'], label=f'alfa={alfa}')
    #log-lin plot
    #plt.yscale('log')
    with open('evaluation_times.json', 'w') as file:
        json.dump(evaluation_times, file)

    plt.xlabel('Iteracje')
    plt.ylabel('Wartość funkcji')
    plt.title('Wykresy zbieżności dla różnych wartości parametru kroku')
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    


#Note that each function takes a 2D array and returns a 1D array
def f3_adapter(x):
    x_reshaped = np.array(x).reshape(1, -1)
    result = f3(x_reshaped)
    return result[0]



alfa_to_test = [1e-10, 1e-9, 1e-8] #[1e-3, 1e-2, 1e-1]
x0_cec = np.random.uniform(-100.0, 100.0, size=(1, 10))


#For qudratic function
plot(f, x0_cec, alfa_to_test)

#For F3 function
plot(f3_adapter, x0_cec, alfa_to_test)