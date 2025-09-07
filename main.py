from Interfaz.interfaz_grafica import CalculadoraOptimizacion
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = CalculadoraOptimizacion(root)
    root.mainloop()