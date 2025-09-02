import time
import sympy as sp
import numpy as np
import scipy.optimize as opt
from .base_optimizer import BaseOptimizer

class UnconstrainedOptimizer(BaseOptimizer):
    """Clase para la optimización sin restricciones"""
    def solve(self, objective, constraints=None, **kwargs):
        """
        Usa scipy.optimize.minimize para optimización sin restricciones
        
        Args:
            objective: Función objetivo
            constraints: Ignorado (método sin restricciones)
        
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()
        
        try:
            obj_expr = self._parser_function(objective)
            variables = self._get_variables(obj_expr)
            
            # Convertir a función numérica
            obj_func = sp.lambdify(variables, obj_expr, 'numpy')
            
            # Punto inicial (centro)
            x0 = np.zeros(len(variables))
            
            # Optimizar usando BFGS
            result = opt.minimize(obj_func, x0, method='BFGS')
            
            # Convertir resultado a diccionario
            optimal_point = {var.name: result.x[i] for i, var in enumerate(variables)}
            optimal_value = float(result.fun)
            
            self.computation_time = time.time() - start_time
            
            return {
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'iterations': result.nit,
                'computation_time': self.computation_time,
                'method': 'Unconstrained (BFGS)'
            }
            
        except Exception as e:
            return {'error': f"Error en optimización sin restricciones: {e}"}