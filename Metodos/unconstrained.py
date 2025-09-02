import time
import sympy as sp
import numpy as np
import scipy.optimize as opt
from .base_optimizer import BaseOptimizer

class UnconstrainedOptimizer(BaseOptimizer):
    """Método general para optimización sin restricciones"""
    
    def solve(self, objective, constraints=None, initial_point=None, **kwargs):
        """
        Usa scipy.optimize.minimize para optimización sin restricciones
        
        Args:
            objective: Función objetivo
            constraints: Ignorado (método sin restricciones)
            initial_point: Punto inicial como diccionario
            **kwargs: Parámetros adicionales
        
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()
        
        try:
            obj_expr = self._parse_function(objective)
            variables = self._get_variables(obj_expr)
            
            # Convertir a función numérica
            obj_func = sp.lambdify(variables, obj_expr, 'numpy')
            
            # Punto inicial
            if initial_point:
                x0 = np.array([initial_point.get(var.name, 0.0) for var in variables])
            else:
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
            return {
                'optimal_point': 'N/A',
                'optimal_value': 'N/A',
                'iterations': 0,
                'computation_time': time.time() - start_time,
                'method': 'Unconstrained (BFGS)',
                'error': f"Error en optimización sin restricciones: {e}"
            }