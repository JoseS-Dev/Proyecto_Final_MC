import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from metodos.metodos_optimización import MetodosOptimizacion

class CalculadoraOptimizacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Optimización No Lineal")
        self.root.geometry("900x500")
        
        # Variables para almacenar entradas
        self.funcion_obj = tk.StringVar()
        self.restricciones = tk.StringVar()
        self.variables = tk.StringVar()
        self.metodo = tk.StringVar(value="sin_restricciones")
        
        # Instancia de la clase de métodos
        self.metodos_opt = MetodosOptimizacion()
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar expansión
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        titulo = ttk.Label(main_frame, text="Calculadora de Optimización No Lineal", 
                          font=("Arial", 16, "bold"))
        titulo.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Entrada de variables
        ttk.Label(main_frame, text="Variables (separadas por coma):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.variables, width=40).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Entrada de función objetivo
        ttk.Label(main_frame, text="Función objetivo:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.funcion_obj, width=40).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Entrada de restricciones
        ttk.Label(main_frame, text="Restricciones (separadas por ;):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.restricciones, width=40).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Método de optimización
        ttk.Label(main_frame, text="Método de optimización:").grid(row=4, column=0, sticky=tk.W, pady=5)
        metodos = ttk.Combobox(main_frame, textvariable=self.metodo, width=37)
        metodos['values'] = (
            'sin_restricciones', 
            'varias_variables_derivacion_parcial',
            'gradiente_descendente',
            'lagrange',
            'con_restricciones',
            'problemas_mas_grandes'
        )
        metodos.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Botones
        botones_frame = ttk.Frame(main_frame)
        botones_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Button(botones_frame, text="Resolver", command=self.resolver).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones_frame, text="Limpiar", command=self.limpiar).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones_frame, text="Graficar", command=self.graficar).pack(side=tk.LEFT, padx=5)
        
        # Área de resultados
        ttk.Label(main_frame, text="Resultados:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.resultados = scrolledtext.ScrolledText(main_frame, width=80, height=20)
        self.resultados.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configurar expansión para el área de resultados
        main_frame.rowconfigure(7, weight=1)
    
    def limpiar(self):
        self.variables.set("")
        self.funcion_obj.set("")
        self.restricciones.set("")
        self.resultados.delete(1.0, tk.END)
    
    def resolver(self):
        try:
            # Obtener valores de entrada
            vars_str = self.variables.get().strip()
            func_str = self.funcion_obj.get().strip()
            restr_str = self.restricciones.get().strip()
            metodo = self.metodo.get()
            
            if not vars_str or not func_str:
                messagebox.showerror("Error", "Debe ingresar variables y función objetivo")
                return
            
            # Procesar variables
            variables = [v.strip() for v in vars_str.split(',')]
            
            # Seleccionar método y resolver
            if metodo == "sin_restricciones":
                resultado = self.metodos_opt.metodo_sin_restricciones(variables, func_str)
            elif metodo == "varias_variables_derivacion_parcial":
                resultado = self.metodos_opt.metodo_derivacion_parcial(variables, func_str)
            elif metodo == "gradiente_descendente":
                resultado = self.metodos_opt.metodo_gradiente_descendente(variables, func_str)
            elif metodo == "lagrange":
                resultado = self.metodos_opt.metodo_lagrange(variables, func_str, restr_str)
            elif metodo == "con_restricciones":
                resultado = self.metodos_opt.metodo_con_restricciones(variables, func_str, restr_str)
            elif metodo == "problemas_mas_grandes":
                resultado = self.metodos_opt.metodo_problemas_mas_grandes(variables, func_str, restr_str)
            else:
                messagebox.showerror("Error", "Método no reconocido")
                return
            
            # Mostrar resultados
            self.mostrar_resultados(resultado)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al resolver: {str(e)}")
    
    def mostrar_resultados(self, resultado):
        self.resultados.delete(1.0, tk.END)
        self.resultados.insert(tk.END, "=== RESULTADOS DE OPTIMIZACIÓN ===\n\n")
        
        for key, value in resultado.items():
            self.resultados.insert(tk.END, f"{key}: {value}\n")
    
    def graficar(self):
        try:
            # Obtener valores de entrada
            vars_str = self.variables.get().strip()
            func_str = self.funcion_obj.get().strip()
            restr_str = self.restricciones.get().strip()
            
            if not vars_str or not func_str:
                messagebox.showerror("Error", "Debe ingresar variables y función objetivo")
                return
            
            # Procesar variables
            variables = [v.strip() for v in vars_str.split(',')]
            
            if len(variables) == 1:
                self.metodos_opt.graficar_2d(variables[0], func_str)
            elif len(variables) == 2:
                self.metodos_opt.graficar_3d(variables, func_str, restr_str)
            else:
                messagebox.showwarning("Advertencia", "Solo se pueden graficar funciones de 1 o 2 variables")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al graficar: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalculadoraOptimizacion(root)
    root.mainloop()