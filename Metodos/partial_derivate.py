import time
import sympy as sp
from .base_optimizer import BaseOptimizer

class PartialDerivativeOptimizer(BaseOptimizer):
    """Optimización usando derivadas parciales (puntos críticos)"""
    
    def solve(self, objective, constraints=None, initial_point=None, **kwargs):
        """
        Encuentra puntos críticos resolviendo ∇f = 0
        
        Args:
            objective: Función objetivo
            constraints: Restricciones (no usado)
            initial_point: Punto inicial (no usado)
            **kwargs: Parámetros adicionales para consistencia
        
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()
        
        try:
            obj_expr = self._parse_function(objective)
            variables = self._get_variables(obj_expr)
            
            # Calcular derivadas parciales
            partial_derivs = [sp.diff(obj_expr, var) for var in variables]
            
            # Resolver sistema de ecuaciones ∇f = 0
            solutions = sp.solve(partial_derivs, variables, dict=True)
            
            if not solutions:
                return {
                    'optimal_point': 'N/A',
                    'optimal_value': 'N/A',
                    'iterations': 1,
                    'computation_time': time.time() - start_time,
                    'method': 'Partial Derivatives',
                    'error': 'No se encontraron puntos críticos'
                }
            
            # Encontrar mínimo entre puntos críticos
            best_value = float('inf')
            best_point = None
            
            for sol in solutions:
                value = float(obj_expr.subs(sol))
                if value < best_value:
                    best_value = value
                    best_point = sol
            
            self.computation_time = time.time() - start_time
            
            return {
                'optimal_point': best_point,
                'optimal_value': best_value,
                'iterations': 1,
                'computation_time': self.computation_time,
                'method': 'Partial Derivatives'
            }
            
        except Exception as e:
            return {
                'optimal_point': 'N/A',
                'optimal_value': 'N/A',
                'iterations': 0,
                'computation_time': time.time() - start_time,
                'method': 'Partial Derivatives',
                'error': f"Error en método de derivadas parciales: {e}"
            }