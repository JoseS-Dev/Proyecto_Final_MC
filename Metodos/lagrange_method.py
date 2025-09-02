import time
import numpy as np
import sympy as sp
from .base_optimizer import BaseOptimizer

class LagrangeMethod(BaseOptimizer):
    """Clase para el método de optimización para restricciones de igualdad"""

    def solve(self, objective, constraints, **kwargs):
        """
        Resuelve usando multiplicadores de Lagrange
        
        Args:
            objective: Función objetivo
            constraints: Lista de restricciones de igualdad
        
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()

        try:
            # Parseamos la función objetivo
            obj_expr = self._parser_function(objective)
            constraint_exprs = [self._parser_function(c) for c in constraints]
            variables = self._get_variables(obj_expr)

            # Creamos los multiplicadores de lagrange
            lambdas = sp.symbols('lambda0:{}'.format(len(constraints)))

            # Función de Lagrange
            L = obj_expr
            for i, constr in enumerate(constraint_exprs):
                L += lambdas[i] * constr
            
            # Calcular gradiente del Lagrangiano
            all_vars = list(variables) + list(lambdas)
            gradient_eqs = [sp.diff(L, var) for var in all_vars]
            
            # Resolver sistema de ecuaciones
            solutions = sp.solve(gradient_eqs, all_vars, dict=True)
            
            if not solutions:
                return {'error': 'No se encontraron soluciones'}
            
            # Evaluar en cada punto crítico
            best_value = float('inf')
            best_point = None
            
            for sol in solutions:
                # Extraer valores de variables (excluyendo lambdas)
                point = {var: sol[var] for var in variables if var in sol}
                
                # Evaluar función objetivo
                value = float(obj_expr.subs(point))
                
                if value < best_value:
                    best_value = value
                    best_point = point
            
            self.computation_time = time.time() - start_time
            
            return {
                'optimal_point': best_point,
                'optimal_value': best_value,
                'iterations': 1,  # Método analítico
                'computation_time': self.computation_time,
                'method': 'Lagrange Multipliers'
            }
            
        except Exception as e:
            return {'error': f"Error en método de Lagrange: {e}"}
