import numpy as np
import sympy as sp
from scipy.optimize import minimize
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class MetodosOptimizacion:
    @staticmethod
    def metodo_sin_restricciones(variables, func_str):
        if len(variables) != 1:
            raise ValueError("Este método solo funciona con una variable")
        
        # Convertir la función string a una función lambda
        x = sp.symbols(variables[0])
        func_expr = sp.sympify(func_str)
        func_lambda = sp.lambdify(x, func_expr, 'numpy')
        
        # Resolver usando scipy.optimize.minimize
        resultado = minimize(func_lambda, x0=0)
        
        return {
            "Método": "Optimización sin restricciones",
            "Función objetivo": func_str,
            "Variable": variables[0],
            "Valor óptimo de x": resultado.x[0],
            "Mínimo de la función": resultado.fun,
            "Éxito": resultado.success,
            "Número de iteraciones": resultado.nit
        }
    
    @staticmethod
    def metodo_derivacion_parcial(variables, func_str):
        # Crear símbolos para las variables
        sym_vars = sp.symbols(variables)

        # Convertir la función string a una expresión sympy
        func_expr = sp.sympify(func_str)

        # Calcular derivadas parciales
        derivadas = [sp.diff(func_expr, var) for var in sym_vars]

        # Resolver el sistema de ecuaciones
        soluciones = sp.solve(derivadas, sym_vars)

        # Evaluar la función en los puntos críticos
        puntos_criticos = []

        # MANEJO DE DIFERENTES TIPOS DE SOLUCIONES
        if isinstance(soluciones, dict):
            # Si hay una sola solución (devuelve diccionario)
            punto = {str(var): soluciones[var] for var in sym_vars}
            valor_funcion = func_expr.subs(soluciones)
            puntos_criticos.append({"punto": punto, "valor": valor_funcion})

        elif isinstance(soluciones, list):
            # Si hay múltiples soluciones (devuelve lista)
            for sol in soluciones:
                if isinstance(sol, dict):
                    # Solución como diccionario
                    punto = {str(var): sol[var] for var in sym_vars}
                    valor_funcion = func_expr.subs(sol)
                    puntos_criticos.append({"punto": punto, "valor": valor_funcion})
                else:
                    # Solución como tupla
                    punto = {str(var): val for var, val in zip(sym_vars, sol)}
                    valor_funcion = func_expr.subs(punto)
                    puntos_criticos.append({"punto": punto, "valor": valor_funcion})

        return {
            "Método": "Derivación parcial para varias variables",
            "Función objetivo": func_str,
            "Variables": variables,
            "Derivadas parciales": [str(d) for d in derivadas],
            "Puntos críticos": puntos_criticos
        }
    
    @staticmethod
    def metodo_gradiente_descendente(variables, func_str):
        # Crear símbolos para las variables
        sym_vars = sp.symbols(variables)

        # Convertir la función string a una expresión sympy
        func_expr = sp.sympify(func_str)
        func_lambda = sp.lambdify(sym_vars, func_expr, 'numpy')

        # Calcular gradiente (derivadas parciales)
        gradiente = [sp.diff(func_expr, var) for var in sym_vars]
        gradiente_lambda = sp.lambdify(sym_vars, gradiente, 'numpy')

        # Parámetros del gradiente descendente
        alpha = 0.1  # Tamaño del paso
        x_vals = [0] * len(variables)  # Punto inicial [0, 0, ...]
        iteraciones = 1000
        tolerancia = 1e-6

        # Gradiente descendente
        for i in range(iteraciones):
            # Calcular gradiente en el punto actual
            grad = np.array(gradiente_lambda(*x_vals))

            # Actualizar valores
            x_vals = x_vals - alpha * grad

            # Verificar criterio de parada
            if np.linalg.norm(grad) < tolerancia:
                break
            
        # Evaluar función en el punto final
        valor_funcion = func_lambda(*x_vals)

        return {
            "Método": "Gradiente descendente",
            "Función objetivo": func_str,
            "Variables": variables,
            "Valores óptimos": dict(zip(variables, x_vals)),
            "Mínimo de la función": valor_funcion,
            "Iteraciones realizadas": i + 1,
            "Norma del gradiente final": float(np.linalg.norm(grad))
        }
    
    @staticmethod
    def metodo_lagrange(variables, func_str, restr_str):
        if not restr_str:
            raise ValueError("Este método requiere restricciones")
        
        # Crear símbolos para las variables y multiplicador de Lagrange
        sym_vars = sp.symbols(variables)
        λ = sp.symbols('λ')
        
        # Convertir las funciones string a expresiones sympy
        func_expr = sp.sympify(func_str)
        restr_expr = sp.sympify(restr_str)
        
        # Definir la función de Lagrange
        L = func_expr + λ * restr_expr
        
        # Derivar parcialmente
        derivadas = []
        for var in sym_vars:
            derivadas.append(sp.diff(L, var))
        derivadas.append(sp.diff(L, λ))  # Derivada respecto a λ
        
        # Resolver el sistema de ecuaciones CON dict=True
        todas_variables = list(sym_vars) + [λ]
        soluciones = sp.solve(derivadas, todas_variables, dict=True)
        
        # Evaluar la función en las soluciones
        resultados = []
        for sol in soluciones:
            # Extraer valores de la solución
            valores = {}
            for var in sym_vars:
                valores[var] = sol[var]
            λ_val = sol[λ]
            
            # Evaluar función objetivo
            f_opt = func_expr.subs(sol)
            
            resultados.append({
                "variables": {str(var): float(valores[var]) for var in sym_vars},
                "lambda": float(λ_val),
                "f(x)": float(f_opt)
            })
        
        return {
            "Método": "Método de Lagrange",
            "Función objetivo": func_str,
            "Restricción": restr_str,
            "Variables": variables,
            "Ecuaciones del sistema": [str(d) + " = 0" for d in derivadas],
            "Soluciones": resultados
        }
    
    @staticmethod
    def metodo_con_restricciones(variables, func_str, restr_str):
        if not restr_str:
            raise ValueError("Este método requiere restricciones")
        
        # Procesar restricciones
        restricciones_list = [r.strip() for r in restr_str.split(';') if r.strip()]
        
        # Convertir la función y restricciones a formato adecuado para scipy
        sym_vars = sp.symbols(variables)
        func_expr = sp.sympify(func_str)
        func_lambda = sp.lambdify(sym_vars, func_expr, 'numpy')
        
        # Configurar restricciones
        constraints = []
        for restr in restricciones_list:
            restr_expr = sp.sympify(restr)
            # Determinar el tipo de restricción (igualdad o desigualdad)
            if '=' in restr and not ('<' in restr or '>' in restr):
                # Restricción de igualdad
                constraint = {'type': 'eq', 'fun': lambda x: sp.lambdify(sym_vars, restr_expr, 'numpy')(*x)}
            else:
                # Restricción de desigualdad (asumimos <= 0)
                constraint = {'type': 'ineq', 'fun': lambda x: -sp.lambdify(sym_vars, restr_expr, 'numpy')(*x)}
            constraints.append(constraint)
        
        # Punto inicial
        x0 = [0] * len(variables)
        
        # Resolver
        resultado = minimize(func_lambda, x0, constraints=constraints)
        
        return {
            "Método": "Optimización con restricciones",
            "Función objetivo": func_str,
            "Restricciones": restricciones_list,
            "Variables": variables,
            "Valores óptimos": dict(zip(variables, resultado.x)),
            "Mínimo de la función": resultado.fun,
            "Éxito": resultado.success
        }
    
    @staticmethod
    def metodo_problemas_mas_grandes(variables, func_str, restr_str):
        # Este método usa cvxpy para problemas de optimización convexa
        
        # Crear variables de optimización
        cvx_vars = [cp.Variable() for _ in range(len(variables))]
        
        # Convertir la función objetivo
        # Nota: cvxpy tiene su propio lenguaje para expresiones, así que necesitamos
        # mapear la función a su formato
        mapping = {var: cvx_var for var, cvx_var in zip(variables, cvx_vars)}
        
        # Reemplazar variables en la función objetivo
        func_cvx = func_str
        for var in variables:
            func_cvx = func_cvx.replace(var, f'mapping["{var}"]')
        
        # Evaluar la expresión (esto es simplificado y puede necesitar ajustes)
        try:
            objetivo = cp.Minimize(eval(func_cvx))
        except:
            raise ValueError("CVXPY no puede manejar esta función objetivo")
        
        # Procesar restricciones
        constraints = []
        if restr_str:
            restricciones_list = [r.strip() for r in restr_str.split(';') if r.strip()]
            for restr in restricciones_list:
                # Reemplazar variables en la restricción
                restr_cvx = restr
                for var in variables:
                    restr_cvx = restr_cvx.replace(var, f'mapping["{var}"]')
                
                # Evaluar la restricción
                try:
                    constraints.append(eval(restr_cvx))
                except:
                    continue
        
        # Resolver el problema
        problema = cp.Problem(objetivo, constraints)
        problema.solve()
        
        return {
            "Método": "Problemas más grandes (CVXPY)",
            "Función objetivo": func_str,
            "Restricciones": restricciones_list if restr_str else "Ninguna",
            "Variables": variables,
            "Valores óptimos": {var: var_obj.value for var, var_obj in zip(variables, cvx_vars)},
            "Mínimo de la función": problema.value,
            "Estado": problema.status
        }
    
    @staticmethod
    def graficar_2d(variable, func_str):
        # Convertir la función string a una función lambda
        x = sp.symbols(variable)
        func_expr = sp.sympify(func_str)
        func_lambda = sp.lambdify(x, func_expr, 'numpy')
        
        # Generar datos para la gráfica
        x_vals = np.linspace(-10, 10, 400)
        y_vals = func_lambda(x_vals)
        
        # Crear gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f'f({variable}) = {func_str}')
        plt.xlabel(variable)
        plt.ylabel(f'f({variable})')
        plt.title(f'Función objetivo: f({variable}) = {func_str}')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    @staticmethod
    def graficar_3d(variables, func_str, restr_str):
        
        # Convertir la función string a una función lambda
        x, y = sp.symbols(variables)
        func_expr = sp.sympify(func_str)
        func_lambda = sp.lambdify((x, y), func_expr, 'numpy')
        
        # Generar datos para la gráfica
        x_vals = np.linspace(-10, 10, 100)
        y_vals = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = func_lambda(X, Y)
        
        # Crear gráfica 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Graficar superficie
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
        
        # Graficar restricciones si existen
        if restr_str:
            restricciones = [r.strip() for r in restr_str.split(';') if r.strip()]
            for i, restr in enumerate(restricciones):
                try:
                    # Intentar graficar la restricción como una curva de nivel en Z=0
                    restr_expr = sp.sympify(restr)
                    restr_lambda = sp.lambdify((x, y), restr_expr, 'numpy')
                    Z_restr = restr_lambda(X, Y)
                    ax.contour(X, Y, Z_restr, levels=[0], colors=f'C{i+1}', linewidths=2, label=restr)
                except:
                    continue
        
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(f'f({variables[0]}, {variables[1]})')
        ax.set_title(f'Función objetivo: f({variables[0]}, {variables[1]}) = {func_str}')
        plt.legend()
        plt.show()