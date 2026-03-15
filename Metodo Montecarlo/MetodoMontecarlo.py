import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import csv
from sympy import var, lambdify , sympify

def convergencia_montecarlo(expresion_sympy, x_min, x_max, y_min, y_max, valor_real, lista_n):
    
    x_sym = var('x')
    f_num = lambdify(x_sym, expresion_sympy, 'numpy')
    archivo_csv = 'resultados_montecarlo.csv'
    
    resultados = [] # Para guardar datos de la tabla
    
    # Proceso de Simulación y Guardado en CSV
    with open(archivo_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N (Muestras)', 'Valor Real', 'Valor Estimado', 'Error Obtenido'])
        
        for n in lista_n:
            incount = 0
            # Simulación de puntos
            for _ in range(n):
                x_rand = random.uniform(x_min, x_max)
                y_rand = random.uniform(y_min, y_max)
                if y_rand <= f_num(x_rand):
                    incount += 1
            
            # Cálculos
            area_total = (x_max - x_min) * (y_max - y_min)
            valor_estimado = (incount / n) * area_total
            error_abs = abs(valor_estimado - valor_real)
            
            # Guardar datos
            resultados.append((n, valor_real, valor_estimado, error_abs))
            writer.writerow([n, valor_real, valor_estimado, error_abs])

    # Presentación de Resultados en Formato Tabla
    print(f"{'N (Muestras)':>12} | {'Valor Real':>12} | {'Valor Estimado':>14} | {'Error':>12}")
    print("-" * 60)
    for r in resultados:
        print(f"{r[0]:12d} | {r[1]:12.6f} | {r[2]:14.6f} | {r[3]:12.6f}")

    # Gráficas de Comportamiento y Convergencia
    n_vals = [r[0] for r in resultados]
    est_vals = [r[2] for r in resultados]
    err_vals = [r[3] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convergencia al valor real
    ax1.axhline(y=valor_real, color='r', linestyle='--', label='Valor Real (Teórico)')
    ax1.plot(n_vals, est_vals, marker='o', color='b', label='Estimación Monte Carlo')
    ax1.set_xscale('log') # Escala logarítmica para apreciar mejor el cambio
    ax1.set_title('Convergencia del Valor Estimado')
    ax1.set_xlabel('Número de muestras (N)')
    ax1.set_ylabel('Área calculada')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    #Disminución del Error
    ax2.plot(n_vals, err_vals, marker='s', color='orange', label='Error Absoluto')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Evolución del Error (Escala Log-Log)')
    ax2.set_xlabel('Número de muestras (N)')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.show()

def montecarlo(expresion, x_min, x_max, y_min, y_max, valor_real, n_total):
    
    f_num = lambdify(var('x'), expresion, 'numpy')
    
    # Preparar el archivo CSV
    archivo_csv = 'convergencia.csv'
    csv_file = open(archivo_csv, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Puntos (N)', 'Estimacion', 'Error'])

    # Configuración de la gráfica
    fig, ax = plt.subplots(figsize=(10, 7))
    x_vals = np.linspace(x_min, x_max, 400)
    ax.plot(x_vals, f_num(x_vals), color='black', linewidth=2, zorder=3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Elementos visuales que se actualizarán
    puntos_in = ax.scatter([], [], color='red', s=5, label='Dentro')
    puntos_out = ax.scatter([], [], color='blue', s=5, label='Fuera')
    ax.legend()

    # Variables de control
    data = {'ins': 0, 'total': 0, 'x_in': [], 'y_in': [], 'x_out': [], 'y_out': []}
    area_rect = (x_max - x_min) * (y_max - y_min)

    def update(frame):
        batch_size = max(1, n_total // 100) 
        
        for _ in range(batch_size):
            if data['total'] >= n_total:
                break
                
            x_r, y_r = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
            data['total'] += 1
            
            if y_r <= f_num(x_r):
                data['ins'] += 1
                data['x_in'].append(x_r)
                data['y_in'].append(y_r)
            else:
                data['x_out'].append(x_r)
                data['y_out'].append(y_r)

        # Actualizar visualización
        puntos_in.set_offsets(np.c_[data['x_in'], data['y_in']])
        puntos_out.set_offsets(np.c_[data['x_out'], data['y_out']])
        
        # Calcular estado actual
        estimacion = (data['ins'] / data['total']) * area_rect
        error = abs(estimacion - valor_real)
        
        # Escribir en CSV
        writer.writerow([data['total'], estimacion, error])
        
        ax.set_title(f"N: {data['total']} | Est: {estimacion:.4f} | Error: {error:.4f}")
        return puntos_in, puntos_out

    
    # frames
    ani = FuncAnimation(fig, update, frames=100, interval=50, repeat=False)
    
    plt.show()
    csv_file.close()
    print(f"Animación finalizada. Datos guardados en {archivo_csv}")


if __name__ == "__main__":

    user_input = str(input("\nIngrese la funcion (ejmp x**2 + 1): "))
    expr = sympify(user_input)

    num =  int(input("Ingrese numero de puntos (ejm 100): "))
    actual_value = float(input("Para probar el error ingrese area de la funcion: "))
    

    n_escogidos = [10,100,500, 1000,5000, 10000,50000, 100000] 
    
    convergencia_montecarlo(
        expresion_sympy = expr,
        x_min=0,
        x_max = 2,
        y_min = 0,
        y_max = 2,
        valor_real =actual_value,
        lista_n = n_escogidos
    )
    montecarlo(
        expresion=expr, 
        x_min=0, 
        x_max=2,
        y_min=0,
        y_max=2,
        valor_real=actual_value,
        n_total=num
    )