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

def solver(f: Callable[[np.ndarray], float], x0: np.ndarray, max_iteration: int, a: int, sigma: float = 0.2) -> Tuple[float, float]:
    iteration = 1
    success_counter = 0
    function_value = f(x0)
    x = x0

    while iteration <= max_iteration:
        mutation = x + sigma * np.random.normal(0, 1, size=x.shape)
        new_function_value = f(mutation)

        if new_function_value <= function_value:
            success_counter += 1
            function_value = new_function_value
            x = mutation
        
        if iteration % a == 0:
            if success_counter / a > 0.2:
                sigma *= 1.22
            if success_counter / a < 0.2:
                sigma *= 0.82
            success_counter = 0
        iteration += 1
    
    return [x, function_value]





