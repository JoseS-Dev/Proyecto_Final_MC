# 📊 Calculadora de optimización No Lineal
Una herramienta computacional desarrollada en python para resolver problemas de optimización no lineal medainte diversos métodos matemáticos

# 🚀 Caracteristicas
- **Multiples metodos de optimización**
- Optimización sin restricciones
- Derivación parcial para varias variables
- Gradiente descendente
- método de Lagrange (restricciones de igualdad)
- **Interfaz gráfica intitiva con TKinter**
- **Visualización 2D y 3D de funciones y restricciones**
- **Manejo flexible de expresiones matemáticas**
- **Resultados detallados con valores óptimos y información del proceso**
# 🗃️ Instalación
- Python 3.11 o superior
- pip (gestor de paquetes de Python)

## Instalar dependencias
```bash
# Clonar o descargar el proyecto
git clone <url-del-repositorio>
cd Proyecto_Final_MC
# Instalar las dependencias
pip install -r requirements.txt
```
## Dependencias principales
- **numpy** - Cálculos numéricos
- **sympy** - Matemáticas simbólicas
- **scipy** - Optimización Cientifica
- **maplotlib** - Visualización Gráfica
- **cvxpy** - Optimización convexa
- **TKinter** - Interfaz Gráfica (Esta incluida en python)

# ♦️Uso
```bash
python main.py
```
## Ejemplos de Entrada
- Optimización sin restricciones (una variable)
```text
Variables: x
Función objetivo: x**2 + 4*x + 6
Restricciones: (dejar vacío)
Método: sin_restricciones
```
- Metódos de Lagrange
```text
Variables: x
Función objetivo: x**2 + 4*x + 6
Restricciones: (dejar vacío)
Método: sin_restricciones
```
- Optimización con restricciones
```text
Variables: x, y
Función objetivo: x**2 + y**2
Restricciones: x + y - 1 = 0
Método: con_restricciones
```
### Sintaxis de Funciones
- Operadores: +, -, *, /, ** (potencia)
- Funciones matemáticas: sin(x), cos(x), exp(x), log(x), sqrt(x)
- Variables: Separadas por comas (ej: x,y,z)
- Restricciones múltiples: Separadas por punto y coma


# 📁 Estructura del proyecto
```text
Proyecto_Final_MC/
│
├── main.py                 # Punto de entrada principal
├── Interfaz_grafica        # Carpeta donde se encuentra la Interfaz
  |── interfaz_grafica.py   # Interfaz de usuario con TKinter
├── metodos      # Carpeta donde se encuentra los metodos de optimización
  |── metodos_optimización.py # Metodos de optimización no lineal
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

# 📊 Visualización
La calculadora incluye capacidades de graficación:
- **Funciones 2D:** Para problemas con una variable
- **Superficies 3D: ** Para problemas con dos variables
- **Restricciones:** Visualizadas como curvas de nivel
# 💻 Programador
- **JoseS-Dev** <a href="https://github.com/JoseS-Dev">✅ Link del perfil</a>

# 🙏 Agradecimientos
Desarrollado como proyecto en la materia **Metodos Cuantitativos** para la Universidad José Antonio Páez, Facultad de Ingeniería, Escuela de Ingeniería en Computación.


