import sympy as sp
import re

class FunctionParser:
    """Utilidades para parsear y manipular expresiones matematicas"""

    @staticmethod
    def parse_function(func_str):
        """Convierte string a expresión sympy con manejo de errores"""
        try:
            # Reemplazar notaciones comunes
            func_str = func_str.replace('^', '**')
            func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)  # 2x -> 2*x
            func_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', func_str)  # x2 -> x*2
            
            return sp.sympify(func_str)
        except Exception as e:
            raise ValueError(f"Error parseando función: {e}")
    
    @staticmethod
    def extract_variables(func_str):
        """Extrae variables de una función string"""
        # Encuentra todos los identificadores válidos (variables)
        variables = set(re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', func_str))
        # Excluir funciones matemáticas comunes
        math_funcs = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs'}
        return sorted([v for v in variables if v not in math_funcs])
    
    @staticmethod
    def validate_constraint(constraint_str):
        """Valida que una restricción sea de igualdad"""
        if '==' not in constraint_str:
            raise ValueError("Las restricciones deben usar '==' para igualdad")
        return constraint_str