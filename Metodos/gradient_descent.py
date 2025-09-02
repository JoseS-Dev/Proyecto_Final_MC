import time
import sympy as sp
import numpy as np
from .base_optimizer import BaseOptimizer

class GradientDescent(BaseOptimizer):
    """Clase para el método de optimización por descenso de gradiente"""
    
    def solve(self, objective, constraints=None, initial_point=None,
              learning_rate=0.01, max_iterations=1000, tolerance=1e-6, **kwargs):
        """
        Resuelve el problema de minimización usando gradiente descendente
        
        Args:
            objective: Función objetivo como string
            constraints: Lista de restricciones (no usado en este método básico)
            initial_point: Punto inicial como diccionario {variable: valor}
            learning_rate: Tasa de aprendizaje
            max_iterations: Máximo número de iteraciones
            tolerance: Tolerancia para convergencia
            **kwargs: Parámetros adicionales para consistencia
        
        Returns:
            Diccionario con resultados
        """
        start_time = time.time()

        try:
            # Parsear la función objetivo
            obj_expr = self._parse_function(objective)
            variables = self._get_variables(obj_expr)

            # Convertir el punto inicial a un array de numpy
            if not initial_point:
                initial_point = {var.name: 0.0 for var in variables}
            
            x_current = np.array([initial_point.get(var.name, 0.0) for var in variables])

            # Calcular el gradiente
            gradient_expr = self._gradient(obj_expr, variables)
            grad_func = sp.lambdify(variables, gradient_expr, "numpy")

            # Iteraciones del gradiente descendente
            for iteration in range(max_iterations):
                # Evaluar gradiente en punto actual
                grad_val = np.array(grad_func(*x_current))
                
                # Actualizar punto
                x_new = x_current - learning_rate * grad_val
                
                # Verificar convergencia
                if np.linalg.norm(x_new - x_current) < tolerance:
                    break
                
                x_current = x_new
                self.iterations = iteration + 1
            
            # Resultados finales
            optimal_point = {var.name: x_current[i] for i, var in enumerate(variables)}
            optimal_value = float(obj_expr.subs(optimal_point))
            
            self.computation_time = time.time() - start_time
            
            return {
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'iterations': self.iterations,
                'computation_time': self.computation_time,
                'method': 'Gradient Descent'
            }
            
        except Exception as e:
            return {
                'optimal_point': 'N/A',
                'optimal_value': 'N/A',
                'iterations': 0,
                'computation_time': time.time() - start_time,
                'method': 'Gradient Descent',
                'error': f"Error en gradiente descendente: {e}"
            }


        