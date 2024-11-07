############## Polecenie ##############

'''
Zaimplementuj algorytm ewolucyjny (1+1)-ES z adaptacją zasięgu mutacji zgodnie z regułą 1/5-sukcesu. 
Zbadaj zaimplementowany solver na nastepujących funkcjach celu: kwadratowej, F3/CEC2017 oraz F7/CEC2017. 
W ramach badania sprawdź jaki wpływ na działanie algorytmu mają parametry sterujące regułą adaptacji zasięgu mutacji oraz początkowa wartość zasięgu mutacji.
Porównaj działanie algorytmu z własną lub dostarczaną w ramach pakietów numerycznych implementacją algorytmu SGD. 
Pamiętaj, aby w badaniach skompensować stochastyczność solwera ewolucyjnego. 
Rozważ dwie wymiarowości zadania optymalizacji (liczba zmiennych decyzyjnych): 10 oraz 30
'''

############## Implementacja ##############


import numpy as np
from cec2017.functions import f3, f7
from typing import Tuple, Callable
from time import time
import matplotlib.pyplot as plt

def solver(f: Callable[[np.ndarray], float], x0: np.ndarray, max_iteration: int = 3000, a: int = 5, sigma: float = 0.2) -> Tuple[float, float]:
    iteration = 1
    success_counter = 0
    function_value = f(x0)
    x = x0
    history = [function_value]
    

    while iteration <= max_iteration:
        mutation = x + sigma * np.random.normal(0, 1, size=x.shape)
        new_function_value = f(mutation)

        if new_function_value <= function_value:
            success_counter += 1
            function_value = new_function_value
            x = mutation
            history.append(new_function_value)
            
        
        if iteration % a == 0:
            if success_counter / a > 0.2:
                sigma *= 1.22
            if success_counter / a < 0.2:
                sigma *= 0.82
            success_counter = 0
        iteration += 1
    
    return x, function_value, history


def plot(f: Callable[[np.ndarray], float], x0: np.ndarray):
    start_time = time()
    x_min, min_value, history = solver(f, x0)
    end_time = time()
    evaluation_time = round(end_time - start_time, 3)
    
    with open('details.txt', 'a') as file:
        file.write(f"Minimum: {x_min}")
        file.write(f"Minimum value: {min_value}")
        file.write(f"Evaluation time: {evaluation_time}")
        file.write("\n" + "################" + "\n")
    
    plt.plot(history)
    plt.xlabel("Iteracje")
    plt.ylabel("Wartość funkcji")
    plt.grid(True)
    plt.show()

def plot_sigmas(f: Callable[[np.ndarray], float], x0: np.ndarray, sigmas: np.ndarray, file_name: str):
    for sigma in sigmas:
        _, _, history = solver(f, x0, sigma=sigma)
        plt.plot(history, label=f'Sigma={sigma}')
    
    plt.xlabel("Iteracje")
    plt.ylabel("Wartość funkcji")
    plt.title("Zbieżność dla różnych wartości sigmy")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

def plot_a(f: Callable[[np.ndarray], float], x0: np.ndarray, a_list: np.ndarray = [0.5, 5, 50], file_name: str = ""):
    for a in a_list:
        _, _, history = solver(f, x0, a=a)
        plt.plot(history, label=f'Paramter a={a}')
    plt.xlabel("Iteracje")
    plt.ylabel("Wartość funkcji")
    plt.title("Zbieżność dla różnych wartości parametru a")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()


def f(x):
    return np.sum(x ** 2)

def f3_adapter(x):
    x_reshaped = np.array(x).reshape(1, -1)
    result = f3(x_reshaped)
    return result[0]

def f7_adapter(x):
    x_reshaped = np.array(x).reshape(1, -1)
    result = f7(x_reshaped)
    return result[0]


x0 = np.random.uniform(-100.0, 100.0, size=(1, 10))
sigmas = [0.02, 0.2, 2]


plot_sigmas(f, x0, sigmas = [0.02, 0.2, 2], file_name="sigma_kwadratowa.png")
plot_sigmas(f3_adapter, x0, sigmas = [0.02, 0.2, 2], file_name="sigma_f3.png")
plot_sigmas(f7_adapter, x0, sigmas = [0.02, 0.2, 2], file_name="sigma_f7.png")

plot_a(f, x0, file_name="a_kwadratowa.png")
plot_a(f3_adapter, x0, file_name="a_f3.png")
plot_a(f7_adapter, x0, file_name="a_f7.png")

    



