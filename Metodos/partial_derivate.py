import time
import sympy as sp
from .base_optimizer import BaseOptimizer

class PartialDerivateOptimizer(BaseOptimizer):
    """Clase para la optimización usando derivadas parciales"""

    def solve(self, objective, constraints=None, **kwargs):
        """
        Encuentra puntos críticos resolviendo ∇f = 0
        
        Args:
            objective: Función objetivo
            constraints: Restricciones (no usado)
        
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()
        try:
            # Parseamos la función objetivo
            obj_expr = self._parser_function(objective)
            variables = self._get_variables(obj_expr)

            # Calculamos las derivadas parciales
            partial_derivs = [sp.diff(obj_expr, var) for var in variables]

            # Resolvemos sistema de ecuaciones
            solutions = sp.solve(partial_derivs, variables, dict=True)

            if not solutions:
                return {'error': 'No se encontraron soluciones'}
            
            # Encotramos minimo entre puntos criticos
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
                'iterations': 1, # Es un metodo Analitico
                'computation_time': self.computation_time,
                'method': 'Partial Derivatives'
            }
            
        except Exception as e:
            return {'error': f"Error en método de derivadas parciales: {e}"}