import time
import sympy as sp
import numpy as np
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """Clase Abstracta para todos los metodos de Optimización"""

    def __init__(self):
        self.computation_time = 0
        self.iterations = 0
    
    @abstractmethod
    def solve(self, objective, constraints=None, **kwargs):
        """Metodo para resolver el problema de optimización"""
        pass

    def _parser_function(self, func_str):
        """Convierte una cadena de función a expresión sympy"""
        return sp.sympify(func_str)
    
    def _get_variables(self, expression):
        """Obtiene las variables de una expresión"""
        return sorted(expression.free_symbols, key=lambda x: x.name)

    def _evaluate_function(self, expression, point):
        """Evalua una función en un punto dado"""
        return expression.subs(point)

    def _gradient(self, expression, variables):
        """"Calcula el gradiente de una función"""
        return [sp.diff(expression, var) for var in variables]

    def _hessian(self, expression, variables):
        """Calcula la matriz Hessiana de una función"""
        num_vars = len(variables)
        hessian = np.zeros((num_vars, num_vars))
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                hessian[i, j] = sp.diff(sp.diff(expression, var1), var2)
        return hessian