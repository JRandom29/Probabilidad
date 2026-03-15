import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def visualizar_tlc(dist_name, **params):
    """
    Visualiza el Teorema Central del Límite para diferentes distribuciones.
    
    Parámetros:
    - dist_name: 'Bernoulli', 'Exponencial', 'Poisson', 'Binomial Negativa'
    - **params: Parámetros requeridos por scipy.stats (p, mu, n, etc.)
    """
    sns.set_theme(style="whitegrid")
    n_sizes = [1, 5, 30, 100]
    n_samples = 5000  # Número de iteraciones para generar medias
    
    #Configurar la distribución base y calcular media (mu) y desviación (sigma) teóricas
    if dist_name.lower() == 'bernoulli':
        dist = stats.bernoulli(p=params['p'])
        mu, sigma = dist.mean(), dist.std()
    elif dist_name.lower() == 'exponencial':
        dist = stats.expon(scale=1/params['lam'])
        mu, sigma = dist.mean(), dist.std()
    elif dist_name.lower() == 'poisson':
        dist = stats.poisson(mu=params['mu'])
        mu, sigma = dist.mean(), dist.std()
    elif dist_name.lower() == 'binomial negativa':
        dist = stats.nbinom(n=params['n'], p=params['p'])
        mu, sigma = dist.mean(), dist.std()
    else:
        raise ValueError("Distribución no soportada.")

    fig, axes = plt.subplots(4, 2, figsize=(10, 6))
    fig.suptitle(f'Demostración del TLC: Distribución {dist_name}', fontsize=16, fontweight='bold')

    for i, n in enumerate(n_sizes):
        # Generar datos
        data = dist.rvs(size=(n_samples, n))
        sample_means = data.mean(axis=1)
        
        # --- Columna Izquierda ----
        ax_hist = axes[i, 0]
        sns.histplot(sample_means, kde=False, stat="density", ax=ax_hist, color='skyblue', alpha=0.6)
        
        # --- Curva Normal Teórica ----
        x = np.linspace(min(sample_means), max(sample_means), 100)
        y = stats.norm.pdf(x, loc=mu, scale=sigma / np.sqrt(n))
        ax_hist.plot(x, y, color='red', lw=2, label='Normal Teórica')
        
        ax_hist.set_title(f'')
        
        ax_hist.text(0.05, 0.90, f'n = {n}', transform=ax_hist.transAxes, 
                     fontsize=14, fontweight='bold', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax_hist.set_ylabel('')
        if i == 0: ax_hist.legend()

        # --- Columna Derecha ---
        ax_cdf = axes[i, 1]
        sns.ecdfplot(sample_means, ax=ax_cdf, color='darkblue', lw=2)
        ax_cdf.set_title(f'')
        ax_cdf.set_ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# uso:
visualizar_tlc('Exponencial', lam=0.5)
visualizar_tlc('Bernoulli', p=0.3)
visualizar_tlc('Poisson', mu=4)
visualizar_tlc('Binomial Negativa', n=5, p=0.5)