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


import autograd.numpy as np
from cec2017.functions import f3, f7
from typing import Callable
from time import time
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import csv
from autograd import grad



def solver(f: Callable[[np.ndarray], float], x0: np.ndarray, max_iteration: int = 30000, a: int = 5, sigma: float = 5, max_no_improve_counter: int = 1000):
    start_time = time()
    iteration = 1
    success_counter = 0
    function_value = f(x0)
    x = x0
    history = [function_value]
    no_improve_counter = 0
    

    while iteration <= max_iteration:
        mutation = x + sigma * np.random.normal(0, 1, size=x.shape)
        new_function_value = f(mutation)

        if new_function_value <= function_value:
            success_counter += 1
            function_value = new_function_value
            x = mutation
            history.append(new_function_value)
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
    end_time = time()
    evaluation_time = round(end_time - start_time, 3)
    
    return x, function_value, history, evaluation_time

def save_results(file_name: str, sigma:float, a: int, function_value: float, evaluation_time: float):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Sigma", "A", "Function value", "Evaluation time"])
        writer.writerow([sigma, a, function_value, evaluation_time])


def plot_sigmas(f: Callable[[np.ndarray], float], x0: np.ndarray, sigmas: np.ndarray, file_name: str):
    plt.figure(figsize=(12, 8))
    for sigma in sigmas:
        function_values = []
        evaluation_times = []
        history_list = []
        for i in range(10):
            x, function_value, history, evaluation_time = solver(f, x0, sigma=sigma)
            function_values.append(function_value)
            evaluation_times.append(evaluation_time)
            history_list.append(history)
        avg_function_value = np.mean(function_values)
        avg_evaluation_time = np.mean(evaluation_times)
        min_lenght = min(len(hist) for hist in history_list)
        cut_histories = [hist[-min_lenght:] for hist in history_list]
        avg_history = np.mean(cut_histories, axis=0)

        save_results("details.csv", sigma, 5, round(avg_function_value, 3), avg_evaluation_time)
        plt.plot(avg_history, label=f'Sigma={sigma}', linewidth=2)
    
    plt.xlabel("Iteracje", fontsize=20, fontweight='bold')
    plt.ylabel("Wartość funkcji", fontsize=20, fontweight='bold')
    plt.title("Zbieżność dla różnych wartości sigmy", fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()
    plt.close()
    

def plot_a(f: Callable[[np.ndarray], float], x0: np.ndarray, a_list: np.ndarray = [5, 20, 50], file_name: str = ""):
    plt.figure(figsize=(12, 8))
    for a in a_list:
        function_values = []
        evaluation_times = []
        history_list = []
        for i in range(10):
            x, function_value, history, evaluation_time = solver(f, x0, a=a)
            function_values.append(function_value)
            evaluation_times.append(evaluation_time)
            history_list.append(history)
        avg_function_value = np.mean(function_values)
        avg_evaluation_time = np.mean(evaluation_times)
        min_lenght = min(len(hist) for hist in history_list)
        cut_histories = [hist[-min_lenght:] for hist in history_list]
        avg_history = np.mean(cut_histories, axis=0)

        save_results("details.csv", 0.5, a, round(avg_function_value, 3), avg_evaluation_time)
        plt.plot(avg_history, label=f'Paramter a={a}', linewidth=2)
    
    plt.xlabel("Iteracje", fontsize=20, fontweight='bold')
    plt.ylabel("Wartość funkcji", fontsize=20, fontweight='bold')
    plt.title("Zbieżność dla różnych wartości parametru a", fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()
    plt.close()



def f3_adapter(x):
    x_reshaped = np.array(x).reshape(1, -1)
    result = f3(x_reshaped)
    return result[0]

def f7_adapter(x):
    x_reshaped = np.array(x).reshape(1, -1)
    result = f7(x_reshaped)
    return result[0]

def f_quadratic(x):
    return np.sum(x ** 2)

def sgd(f, x0, alfa=1e-8, epsilon=1e-6, iterations=3000):
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
        #print(i)

    
    return f(x), f_values

def plot_sgd_es(x0_10):

    plt.figure(figsize=(12, 8))
    start_time = time()
    _, f_values = sgd(f3_adapter, x0=x0_10)
    end_time = time()
    print(round(end_time - start_time, 3))
    plt.plot(f_values, label='', linewidth=2)
    print(round(end_time - start_time, 3))
    plt.plot(f_values, linewidth=2)
    plt.xlabel("Iteracje", fontsize=20, fontweight='bold')
    plt.ylabel("Wartość funkcji", fontsize=20, fontweight='bold')
    plt.title("SGD - funkcja f3", fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig("sgd_compare_f3.png")
    plt.show()

    plt.close()
    plt.figure(figsize=(12, 8))
    x, function_value, history, evaluation_time = solver(f3_adapter, x0_10)
    print(evaluation_time)
    plt.plot(history, linewidth=2)
    plt.xlabel("Iteracje", fontsize=20, fontweight='bold')
    plt.ylabel("Wartość funkcji", fontsize=20, fontweight='bold')
    plt.title('(1+1)-ES - funkcja f3', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig("es_compare_f3.png")
    plt.show()

def wilcoxon_test(f: Callable[[np.ndarray], float]):
    resultes = []
    resultsgd = []
    for i in range(10):
        x0 = np.random.uniform(-100.0, 100.0, size=(1, 10))
        _, function_value, _, _ = solver(f, x0)
        resultes.append(function_value)

        f_value, _ = sgd(f, x0=x0)
        resultsgd.append(f_value)
        print(i)

    stat, p_values = wilcoxon(resultes, resultsgd)
    print(stat, p_values)
    return stat, p_values


def main():

    x0_10 = np.random.uniform(-100.0, 100.0, size=(1, 10))
    x0_30 = np.random.uniform(-100.0, 100.0, size=(1, 30))
    
    sigmas = [0.1, 2, 5]
    
    '''
    # Eksperymenty dla funkcji kwadratowej
    
    plot_sigmas(f_quadratic, x0_10, sigmas=sigmas, file_name="sigma_kwadratowa_10.png")
    plot_sigmas(f_quadratic, x0_30, sigmas=sigmas, file_name="sigma_kwadratowa_30.png")
    plot_a(f_quadratic, x0_10, file_name="a_kwadratowa_10.png")
    plot_a(f_quadratic, x0_30, file_name="a_kwadratowa_30.png")
    
    # Eksperymenty dla funkcji F3
    
    plot_sigmas(f3_adapter, x0_10, sigmas=sigmas, file_name="sigma_f3_10.png")
    plot_sigmas(f3_adapter, x0_30, sigmas=sigmas, file_name="sigma_f3_30.png")
    plot_a(f3_adapter, x0_10, file_name="a_f3_10.png")
    plot_a(f3_adapter, x0_30, file_name="a_f3_30.png")
    
    # Eksperymenty dla funkcji F7
    plot_sigmas(f7_adapter, x0_10, sigmas=sigmas, file_name="sigma_f7_10.png")
    plot_sigmas(f7_adapter, x0_30, sigmas=sigmas, file_name="sigma_f7_30.png")
    plot_a(f7_adapter, x0_10, file_name="a_f7_10.png")
    plot_a(f7_adapter, x0_30, file_name="a_f7_30.png")
    '''
    #Wilcoxon test
    #wilcoxon_test(f_quadratic)
    #wilcoxon_test(f3_adapter)
    #wilcoxon_test(f7_adapter, x0_10)
    
    
    
    

if __name__=="__main__":
    main()

    



