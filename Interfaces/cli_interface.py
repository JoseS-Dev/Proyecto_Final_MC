import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Clase para la Interface
class CliInterface:
    """Interface de línea de comandos para la visualización de funciones matemáticas."""

    def get_problem_input(self):
        """"Obtiene los datos del problema a resolver."""
        print(" Ingrese lso datos del problema ")

        # Función Objetivo
        func_objetivo = input("Ingrese la función objetivo: ej(x**2 + y**2): ")

        # Restricciones
        restricciones = []
        while True:
            restriccion = input("Ingrese una restricción (o 'fin' para terminar): ")
            if restriccion.lower() == 'fin':
                break
            restricciones.append(restriccion)

        # Punto inicial para metodos iterativos
        punt_inicial = {}
        print("Ingrese punto inicial (variables separadas por coma: ej: x1=1, x2=2):")
        punt = input("Punto inicial: ")
        if punt:
            for par in punt.split(","):
                var, val = par.split("=")
                punt_inicial[var.strip()] = float(val.strip())
        
        # Parametros del método
        tasa_aprendizaje = float(input("Ingrese la tasa de aprendizaje (ej: 0.01): "))
        max_iteración = int(input("Ingrese el número máximo de iteraciones (ej: 1000): "))

        # Metodos a Usar
        print("Todos los metodos")
        print("1. Todos los Métodos")
        print("2. Gradiente Descendente")
        print("3. Método de Lagrange")
        print("4. Derivadas Parciales")
        print("5. Sin restricciones")

        method_opc = input("Seleccione un método (1-5): ")

        method_opcs = {
            '1': "all",
            '2': 'gradient',
            '3': 'lagrange',
            '4': 'partial',
            '5': 'unconstrained'
        }

        return {
            'objective': func_objetivo,
            'constraints': restricciones,
            'initial_point': punt_inicial,
            'learning_rate': tasa_aprendizaje,
            'max_iterations': max_iteración,
            'method': method_opcs.get(method_opc, 'all')
        }
    
    def display_comparative_results(self, results, problem_data):
        """Muestra los resultados comparativos de todos los métodos"""
        print("\n" + "=" * 60)
        print("Resultados Comparativos")
        print("="*60)

        for method, result in results.items():
            if(result):
                print(f"\n {method.upper()}:")
                print(f" Punto óptimo: {result.get('optimal_point', 'N/A')}")
                print(f" Valor óptimo: {result.get('optimal_value'):.6f}")
                print(f" Iteraciones: {result.get('iterations', 'N/A')}")
                print(f" Tiempo: {result.get('computation_time', 'N/A'):.6f} s")

                # Se verifica si el problema trae restricciones
                if problem_data['constraints']:
                    constraint_check = self._check_constraints(
                        result.get('optimal_point', {}),
                        problem_data['constraints']
                    )
                    print(f" Restricciones satisfechas: {constraint_check}")
        
        # Generar las graficas si es posible generar
        self._generate_plots(results, problem_data)
    
    def _check_constraints(self, point, constraints):
        """Verifica si el puntos sastiface las restricciones"""
        if not point or not constraints:
            return "N/A"
        try:
            for constraint in constraints:
                # Evaluamos la restricción en el punto
                expr = constraint.replace('==', '-')
                for var, val in point.items():
                    expr = expr.replace(var, str(val))
                result = eval(expr)
                if abs(result) > 1e-6:
                    return f"no (error: {result})"
            return "sí"
        except Exception as e:
            return f"Error al evaluar restricciones: {e}"
    
    def _generate_plots(self, results, problem_data):
        """Genera Graficas 2d o 3d para visualizar los resultados"""
        try:
            # Extraemos variables de la función objetivo
            fun_obj = problem_data['objective']
            var_obj = sorted(set([c for c in fun_obj if c.isalpha()]))

            if(len(var_obj) == 1):
                # Graficar en 2D
                self._plot_2d(fun_obj, results, var_obj[0])
            elif(len(var_obj) == 2):
                # Graficar en 3D
                self._plot_3d(fun_obj, results, var_obj)
            else:
                print("Demasiadas variables para graficar.")
        except Exception as e:
            print(f"Error al generar graficas: {e}")
    
    def _plot_2d(self, objective, results, variable):
        """Genera una gráfica 2D del resultado"""
        x = sp.symbols(variable)
        f = sp.lambdify(x, objective, 'numpy')

        x_values = np.linspace(-10, 10, 400)
        y_values = f(x_values)

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label=f'f({variable}) = {objective}')
        
        # Marcar puntos óptimos
        for method, result in results.items():
            if result and variable in result.get('optimal_point', {}):
                x_opt = result['optimal_point'][variable]
                y_opt = result['optimal_value']
                plt.scatter(x_opt, y_opt, label=f'{method} (x={x_opt:.3f})')
        
        plt.xlabel(variable)
        plt.ylabel('f(' + variable + ')')
        plt.title('Optimización No Lineal - Resultados')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def _plot_3d(self, objective, results, variables):
        """Genera una Grafica 3d del resultado"""
        x, y = sp.symbols(variables)
        f = sp.lambdify((x,y), objective, 'numpy')

        x_values = np.linspace(-5, 40, 40)
        y_values = np.linspace(-5, 40, 40)
        X, Y = np.meshgrid(x_values, y_values)
        Z = f(X, Y)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
        
        # Marcar puntos óptimos
        for method, result in results.items():
            if result and all(v in result.get('optimal_point', {}) for v in variables):
                x_opt = result['optimal_point'][variables[0]]
                y_opt = result['optimal_point'][variables[1]]
                z_opt = result['optimal_value']
                ax.scatter(x_opt, y_opt, z_opt, color='red', s=100, 
                          label=f'{method} (x={x_opt:.3f}, y={y_opt:.3f})')
        
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel('f(' + ', '.join(variables) + ')')
        ax.set_title('Optimización No Lineal - Superficie 3D')
        ax.legend()
        plt.show()
        