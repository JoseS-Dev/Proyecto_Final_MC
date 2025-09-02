import time
import numpy as np
import sympy as sp
from .base_optimizer import BaseOptimizer

class LagrangeMethod(BaseOptimizer):
    """Implementación del método de Lagrange para restricciones de igualdad"""
    
    def solve(self, objective, constraints=None, initial_point=None, **kwargs):
        """
        Resuelve usando multiplicadores de Lagrange
        
        Args:
            objective: Función objetivo
            constraints: Lista de restricciones de igualdad
            initial_point: Punto inicial (no usado en Lagrange)
            **kwargs: Parámetros adicionales
        
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()
        
        try:
            # Verificar que hay restricciones
            if not constraints:
                return {
                    'optimal_point': 'N/A',
                    'optimal_value': 'N/A',
                    'iterations': 0,
                    'computation_time': time.time() - start_time,
                    'method': 'Lagrange Multipliers',
                    'error': 'Método de Lagrange requiere restricciones'
                }
            
            # Parsear función objetivo y restricciones
            obj_expr = self._parse_function(objective)
            constraint_exprs = [self._parse_function(constr.replace('==', '-')) for constr in constraints]
            
            variables = self._get_variables(obj_expr)
            
            # Crear multiplicadores de Lagrange
            num_constraints = len(constraints)
            lambdas = sp.symbols(f'lambda0:{num_constraints}')
            
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
                return {
                    'optimal_point': 'N/A',
                    'optimal_value': 'N/A',
                    'iterations': 1,
                    'computation_time': time.time() - start_time,
                    'method': 'Lagrange Multipliers',
                    'error': 'No se encontraron soluciones al sistema de ecuaciones'
                }
            
            # Evaluar en cada punto crítico
            best_value = float('inf')
            best_point = None
            
            for sol in solutions:
                # Extraer valores de variables (excluyendo lambdas)
                point = {var: sol[var] for var in variables if var in sol}
                
                # Verificar que todas las variables tengan valores
                if len(point) == len(variables):
                    # Evaluar función objetivo
                    value = float(obj_expr.subs(point))
                    
                    if value < best_value:
                        best_value = value
                        best_point = point
            
            if best_point is None:
                return {
                    'optimal_point': 'N/A',
                    'optimal_value': 'N/A',
                    'iterations': 1,
                    'computation_time': time.time() - start_time,
                    'method': 'Lagrange Multipliers',
                    'error': 'No se encontró un punto óptimo válido'
                }
            
            self.computation_time = time.time() - start_time
            
            return {
                'optimal_point': best_point,
                'optimal_value': best_value,
                'iterations': 1,
                'computation_time': self.computation_time,
                'method': 'Lagrange Multipliers'
            }
            
        except Exception as e:
            return {
                'optimal_point': 'N/A',
                'optimal_value': 'N/A',
                'iterations': 0,
                'computation_time': time.time() - start_time,
                'method': 'Lagrange Multipliers',
                'error': f"Error en método de Lagrange: {str(e)}"
            }
