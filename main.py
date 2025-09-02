from Interfaces.cli_interface import CliInterface
from Metodos.lagrange_method import LagrangeMethod
from Metodos.gradient_descent import GradientDescent
from Metodos.partial_derivate import PartialDerivativeOptimizer
from Metodos.unconstrained import UnconstrainedOptimizer

def main():
    """Función principal del programa"""
    print("=== Calculadora de Optimización No Lineal ===")
    print("Universidad José Antonio Páez")
    print("Facultad de Ingeniería - Escuela de Ingeniería en Computación")
    print("=" * 50)
    
    # Crear interfaz
    interface = CliInterface()
    
    # Obtener problema del usuario
    problem_data = interface.get_problem_input()
    
    # Parámetros comunes para todos los métodos
    common_params = {
        'objective': problem_data['objective'],
        'constraints': problem_data['constraints'],
        'initial_point': problem_data['initial_point'],
        'learning_rate': problem_data.get('learning_rate', 0.01),
        'max_iterations': problem_data.get('max_iterations', 1000),
        'tolerance': problem_data.get('tolerance', 1e-6)
    }
    
    # Resolver con diferentes métodos
    results = {}
    
    if problem_data['method'] == 'all' or problem_data['method'] == 'gradient':
        print("\n--- Resolviendo con Gradiente Descendente ---")
        gradient_solver = GradientDescent()
        results['gradient'] = gradient_solver.solve(**common_params)
    
    if problem_data['method'] == 'all' or problem_data['method'] == 'lagrange':
        print("\n--- Resolviendo con Método de Lagrange ---")
        lagrange_solver = LagrangeMethod()
        results['lagrange'] = lagrange_solver.solve(**common_params)
    
    if problem_data['method'] == 'all' or problem_data['method'] == 'partial':
        print("\n--- Resolviendo con Derivadas Parciales ---")
        partial_solver = PartialDerivativeOptimizer()
        results['partial'] = partial_solver.solve(**common_params)
    
    if problem_data['method'] == 'all' or problem_data['method'] == 'unconstrained':
        print("\n--- Resolviendo con Método Sin Restricciones ---")
        unconstrained_solver = UnconstrainedOptimizer()
        results['unconstrained'] = unconstrained_solver.solve(**common_params)
    
    # Mostrar resultados comparativos
    interface.display_comparative_results(results, problem_data)

if __name__ == "__main__":
    main()