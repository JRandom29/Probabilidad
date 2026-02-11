import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def graficar_uniforme_discreta(valores_n):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    for n in valores_n:
        x = np.arange(1, n + 1)
        y = np.full(n, 1/n)
        y_acumulada = np.cumsum(y)

        ax1.scatter(x, y, marker='o', s=30, label=f'n = {n} (P={1/n:.3f})', alpha=0.8)
        ax2.step(x, y_acumulada, where='post', label=f'n = {n}', linewidth=1.5)

    ax1.set_title('Función de Masa de Probabilidad')
    ax1.set_xlabel('Número de éxitos')
    ax1.set_ylabel('Probabilidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_bernoulli(valores_n):
    valores_n.sort()
    plt.figure(figsize=(10, 6))
    
    x = np.array([0, 1])
    n_series = len(valores_n)
    ancho_barra = 0.8 / n_series
    
    for i, p in enumerate(valores_n):
        offset = (i - (n_series - 1) / 2) * ancho_barra
        
        y = [1 - p, p]
        plt.bar(x + offset, y, width=ancho_barra, alpha=0.8, label=f'p={p}')

    # Configuración de ejes
    plt.xticks(x, ['0 (Fracaso)', '1 (Éxito)']) # Etiquetas claras para Bernoulli
    plt.yticks(np.arange(0, 1.1, 0.1)) # La probabilidad no pasa de 1.0
    
    plt.xlabel('Resultado (x)')
    plt.ylabel('Probabilidad $P(X=x)$')
    plt.title('Comparación de Distribuciones Bernoulli', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(title="Probabilidad de éxito")
    
    plt.show()

def graficar_binomial(valores_n, valores_p):
    # Creamos una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Usamos zip para recorrer ambos vectores al mismo tiempo
    for n, p in zip(valores_n, valores_p):
        # Definir la distribución y el rango x
        dist = stats.binom(n, p)
        
        x = np.arange(0, max(valores_n) + 1)
        
        # Cálculos
        pmf = dist.pmf(x)
        cdf = dist.cdf(x)
        
        label = f'n={n}, p={p}'
        ax1.plot(x, pmf, '-o', alpha=0.6, label=label)
        # Graficar CDF Distribución Acumulada
        ax2.step(x, cdf, where='post', alpha=0.7, label=label)

    # Configuración de estética para PMF
    ax1.set_title('Función de Masa de Probabilidad')
    ax1.set_xlabel('Número de éxitos')
    ax1.set_ylabel('Probabilidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def graficar_geometrica(valores_p):
    # Creamos una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Usamos zip para recorrer ambos vectores al mismo tiempo
    for p in valores_p:
        # Definir la distribución y el rango x
        dist = stats.geom(p)
        
        x = np.arange(dist.ppf(0.01),  dist.ppf(0.99))
        
        # Cálculos
        pmf = dist.pmf(x)
        cdf = dist.cdf(x)
        
        # Graficar  Masa de Probabilidad
        labels = f'p={p}'
        ax1.plot(x, pmf, '-o', alpha=0.6, label=labels)
        # Graficar CDF (Distribución Acumulada)
        ax2.step(x, cdf, where='post', alpha=0.7, label=labels)

    # Configuración de estética para PMF
    ax1.set_title('Función de Masa de Probabilidad')
    ax1.set_xlabel('Número de éxitos')
    ax1.set_ylabel('Probabilidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def graficar_binomial_negativa(valores_r, valores_p):
    # Creamos una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Usamos zip para recorrer ambos vectores al mismo tiempo
    for r, p in zip(valores_r, valores_p):
        # Definir la distribución y el rango x
        dist = stats.nbinom(r, p)
        
        x = np.arange(0, max(valores_r) + 1)
        
        # Cálculos
        pmf = dist.pmf(x)
        cdf = dist.cdf(x)
        
        # Graficar  Masa de Probabilidad
        label = f'r={r}, p={p}'
        ax1.plot(x, pmf, '-o', alpha=0.6, label=label)
        # Graficar CDF (Distribución Acumulada)
        ax2.step(x, cdf, where='post', alpha=0.7, label=label)

    # Configuración de estética para PMF
    ax1.set_title('Función de Masa de Probabilidad')
    ax1.set_xlabel('Número de éxitos')
    ax1.set_ylabel('Probabilidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_hipergeometrica(valores_N, valores_n,valores_K):
    # Creamos una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Usamos zip para recorrer ambos vectores al mismo tiempo
    for N, n, K in zip(valores_N, valores_n, valores_K):
        # Definir la distribución y el rango x
        dist = stats.hypergeom(N, K, n)
        
        x = np.arange(0, min(n, K) + 1)
        
        # Cálculos
        pmf = dist.pmf(x)
        cdf = dist.cdf(x)
        
        # Graficar  Masa de Probabilidad
        labels = f'N={N}, n={n}, K={K}'
        ax1.plot(x, pmf, '-o', alpha=0.6, label=labels)
        # Graficar CDF (Distribución Acumulada)
        ax2.step(x, cdf, where='post', alpha=0.7, label=labels)

    # Configuración de estética para PMF
    ax1.set_title('Función de Masa de Probabilidad')
    ax1.set_xlabel('Número de éxitos')
    ax1.set_ylabel('Probabilidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def graficar_poisson(valores_lamba):
    # Creamos una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Usamos zip para recorrer ambos vectores al mismo tiempo
    for lambda_val in valores_lamba:
        # Definir la distribución y el rango x
        dist = stats.poisson(lambda_val)
        x = np.arange(0, 20)  # Ajustar el rango según necesidad
        
        # Cálculos
        pmf = dist.pmf(x)
        cdf = dist.cdf(x)
        
        # Graficar  Masa de Probabilidad
        label = f'λ={lambda_val}'
        ax1.plot(x, pmf, '-o', alpha=0.6, label=label)
        # Graficar CDF (Distribución Acumulada)
        ax2.step(x, cdf, where='post', alpha=0.7, label=label)

    # Configuración de estética para PMF
    ax1.set_title('Función de Masa de Probabilidad')
    ax1.set_xlabel('Número de éxitos')
    ax1.set_ylabel('Probabilidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_uniforme_continua(valores_a, valores_b):
    # Creamos una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Usamos zip para recorrer ambos vectores al mismo tiempo
    for a, b in zip(valores_a, valores_b):
        # Definir la distribución y el rango x
        dist = stats.uniform(a, b-a)
        x = np.linspace(a, b, 100)
        
        # Cálculos
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        
        # Graficar PDF (Densidad de Probabilidad
        label = f'a={a}, b={b}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        # Graficar CDF (Distribución Acumulada)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)

    # Configuración de estética para PDF
    ax1.set_title('Función de Densidad de Probabilidad')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_exponencial(valores_lambda):
    # Creamos una figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Usamos zip para recorrer ambos vectores al mismo tiempo
    for lambda_val in valores_lambda:
        # Definir la distribución y el rango x
        dist = stats.expon(scale=1/lambda_val)
        x = np.linspace(0, 2,100)  # Ajustar el rango según necesidad
        
        # Cálculos
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        
        # Graficar PDF (Densidad de Probabilidad
        label = f'λ={lambda_val}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        # Graficar CDF (Distribución Acumulada)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)

    # Configuración de estética para PDF
    ax1.set_title('Función de Densidad de Probabilidad')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # Configuración de estética para CDF
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_gamma(valores_alfa, valores_lambda):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for alfa, lambda_val in zip(valores_alfa, valores_lambda):
        dist = stats.gamma(alfa, scale=1/lambda_val)
        x = np.linspace(max(0, dist.ppf(0.01)), dist.ppf(0.99), 200)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        label = f'k={alfa}, θ={1/lambda_val:.2f}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)
    ax1.set_title('Función de Densidad de Probabilidad (Gamma)')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_title('Función de Distribución Acumulada (Gamma)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_beta(valores_a, valores_b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for a, b in zip(valores_a, valores_b):
        dist = stats.beta(a, b)
        x = np.linspace(0, 1, 200)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        label = f'a={a}, b={b}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)
    ax1.set_title('Función de Densidad de Probabilidad (Beta)')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_title('Función de Distribución Acumulada (Beta)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_weibull(valores_k, valores_lambda):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for c, lambda_val in zip(valores_k, valores_lambda):
        dist = stats.weibull_min(c, scale=1/lambda_val  )
        x = np.linspace(max(0, dist.ppf(0.01)), dist.ppf(0.99), 200)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        label = f'k={c}, λ={lambda_val}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)
    ax1.set_title('Función de Densidad de Probabilidad (Weibull)')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_title('Función de Distribución Acumulada (Weibull)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_normal(valores_mu, valores_sigma):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for mu, sigma in zip(valores_mu, valores_sigma):
        dist = stats.norm(loc=mu, scale=sigma)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        label = f'μ={mu}, σ={sigma}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)
    ax1.set_title('Función de Densidad de Probabilidad (Normal)')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_title('Función de Distribución Acumulada (Normal)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_chi2(valores_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for df in valores_df:
        dist = stats.chi2(df)
        x = np.linspace(max(0, dist.ppf(0.01)), dist.ppf(0.99), 200)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        label = f'df={df}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)
    ax1.set_title('Función de Densidad de Probabilidad (Ji-cuadrado)')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_title('Función de Distribución Acumulada (Ji-cuadrado)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_t(valores_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for df in valores_df:
        dist = stats.t(df)
        x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 300)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        label = f'df={df}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)
    ax1.set_title('Función de Densidad de Probabilidad (t student)')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_title('Función de Distribución Acumulada (t student)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_f(valores_df1, valores_df2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for d1, d2 in zip(valores_df1, valores_df2):
        dist = stats.f(d1, d2)
        x = np.linspace(max(0, dist.ppf(0.01)), dist.ppf(0.99), 300)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        label = f'd1={d1}, d2={d2}'
        ax1.plot(x, pdf, '-', alpha=0.6, label=label)
        ax2.plot(x, cdf, '-', alpha=0.7, label=label)
    ax1.set_title('Función de Densidad de Probabilidad (F)')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_title('Función de Distribución Acumulada (F)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# Graficadora menu
opcion = 0
while opcion != 17:
    
    print("Seleccione la distribución a graficar:")
    print("1. Distribución Uniforme Discreta")
    print("2. Distribución Bernoulli")
    print("3. Distribución Binomial")
    print("4. Distribución Geométrica")
    print("5. Distribución binomial negativa")
    print("6. Distribución hipergeométrica")
    print("7. Distribución poisson")
    print("8. Distribución uniforme continua")
    print("9. Distribución exponencial")
    print("10. Distribución gamma")
    print("11. Distribución beta")
    print("12. Distribución wibull")
    print("13. Distribución normal")
    print("14. Distribución ji-cuadrado")
    print("15. Distribución t (student)")
    print("16. Distribución F (fisher)")
    print("17. Salir")

    opcion = int(input("Ingrese el número de la opción deseada: "))
    
    if opcion == 1:
        valores_n = input("Ingrese los valores de n separados por comas (ejemplo: 4,10,40): ")
        valores_n = [int(n.strip()) for n in valores_n.split(',')]
        graficar_uniforme_discreta(valores_n)
    
    if opcion == 2:
        p = input("Ingrese los valores de p separados por comas (ejemplo: 0.2,0.5,0.8): ")
        valores_p = [float(p.strip()) for p in p.split(',')]
        graficar_bernoulli(valores_p)

    if opcion == 3:
        n = input("Ingrese los valores de n separados por comas (ejemplo: 10,20,30): ")
        p = input("Ingrese los valores de p separados por comas (ejemplo: 0.2,0.5,0.8): ")
        valores_n = [int(n.strip()) for n in n.split(',')]
        valores_p = [float(p.strip()) for p in p.split(',')]
        graficar_binomial(valores_n, valores_p)
    
    if opcion == 4:
        p = input("Ingrese los valores de p separados por comas (ejemplo: 0.2,0.5,0.8): ")
        valores_p = [float(p.strip()) for p in p.split(',')]
        graficar_geometrica(valores_p)
    
    if opcion == 5:
        r = input("Ingrese los valores de r separados por comas (ejemplo: 10,20,30): ")
        p = input("Ingrese los valores de p separados por comas (ejemplo: 0.2,0.5,0.8): ")
        valores_r = [int(r.strip()) for r in r.split(',')]
        valores_p = [float(p.strip()) for p in p.split(',')]
        graficar_binomial_negativa(valores_r, valores_p)
    
    if opcion == 6:
        N = input("Ingrese los valores de N separados por comas (ejemplo: 10,20,30): ")
        n = input("Ingrese los valores de n separados por comas (ejemplo: 5,7,3): ")
        K = input("Ingrese los valores de K separados por comas (ejemplo: 5,10,15): ")
        valores_N = [int(N.strip()) for N in N.split(',')]
        valores_n = [int(n.strip()) for n in n.split(',')]
        valores_K = [int(K.strip()) for K in K.split(',')]
        graficar_hipergeometrica(valores_N, valores_n, valores_K)
    
    if opcion == 7:
        lamba_val = input("Ingrese los valores de lambda separados por comas (ejemplo: 10,20,30): ")
        valores_lamba = [int(n.strip()) for n in lamba_val.split(',')]
        graficar_poisson(valores_lamba)
    
    if opcion == 8:
        a = input("Ingrese los valores de a separados por comas (ejemplo: 1,2,3): ")
        b = input("Ingrese los valores de b separados por comas (ejemplo: 4,5,6): ")
        valores_a = [float(a.strip()) for a in a.split(',')]
        valores_b = [float(b.strip()) for b in b.split(',')]
        graficar_uniforme_continua(valores_a, valores_b)
    
    if opcion == 9:
        lamba_val = input("Ingrese los valores de lambda separados por comas (ejemplo: 10,20,30): ")
        valores_lamba = [int(n.strip()) for n in lamba_val.split(',')]
        graficar_exponencial(valores_lamba)
    
    if opcion == 10:
        alfa = input("Ingrese los valores de alfa separados por comas (ejemplo: 1,2): ")
        lamba_val = input("Ingrese los valores de lambda separados por comas (ejemplo: 1,2): ")
        valores_alfa = [float(v.strip()) for v in alfa.split(',')]
        valores_lambda = [float(v.strip()) for v in lamba_val.split(',')]
        graficar_gamma(valores_alfa, valores_lambda)

    if opcion == 11:
        alfa = input("Ingrese los valores de alpha separados por comas (ejemplo: 0.5,1): ")
        beta = input("Ingrese los valores de beta separados por comas (ejemplo: 0.5,2): ")
        valores_a = [float(v.strip()) for v in alfa.split(',')]
        valores_b = [float(v.strip()) for v in beta.split(',')]
        graficar_beta(valores_a, valores_b)

    if opcion == 12:
        k = input("Ingrese los valores de k separados por comas (ejemplo: 1,1.5): ")
        lambda_val = input("Ingrese los valores de lambda separados por comas (ejemplo: 1,2): ")
        valores_k = [float(v.strip()) for v in k.split(',')]
        valores_lambda = [float(v.strip()) for v in lambda_val.split(',')]
        graficar_weibull(valores_k, valores_lambda)

    if opcion == 13:
        mu = input("Ingrese los valores de μ separados por comas (ejemplo: 0,1): ")
        sigma = input("Ingrese los valores de σ separados por comas (ejemplo: 1,2): ")
        valores_mu = [float(v.strip()) for v in mu.split(',')]
        valores_sigma = [float(v.strip()) for v in sigma.split(',')]
        graficar_normal(valores_mu, valores_sigma)

    if opcion == 14:
        dfs = input("Ingrese los grados de libertad separados por comas (ejemplo: 1,2,5): ")
        valores_df = [float(v.strip()) for v in dfs.split(',')]
        graficar_chi2(valores_df)

    if opcion == 15:
        dfs = input("Ingrese los grados de libertad para t separados por comas (ejemplo: 1,5,10): ")
        valores_df = [float(v.strip()) for v in dfs.split(',')]
        graficar_t(valores_df)

    if opcion == 16:
        d1 = input("Ingrese los grados de libertad d1 separados por comas (ejemplo: 1,5): ")
        d2 = input("Ingrese los grados de libertad d2 separados por comas (ejemplo: 5,10): ")
        valores_d1 = [float(v.strip()) for v in d1.split(',')]
        valores_d2 = [float(v.strip()) for v in d2.split(',')]
        graficar_f(valores_d1, valores_d2)
    
