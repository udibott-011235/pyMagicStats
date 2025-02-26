import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns



def output_format(data=None, lb=None, ub=None, bool_result=None, p_value=None, r2=None, txt=None, h_stat=None):
        #Parameters 
        #lb Lower Bound (Float, optional)
        #ub Upper Bound (Float, optional)
        #bool_result Boolean result (bool, optional)
        #r2 R^2 (float, optional)
    
    if data is not None:
        return data 
    
    output = {}


    if lb is not None and ub is not None:
        output['lb'] = lb
        output['ub'] = ub

    if p_value is not None : 
        output['p_value'] = p_value

    if bool_result is not None:
        output['Result'] = np.bool_(bool_result)
    
    if r2 is not None: 
        output['R^2'] = r2
        
    if txt is not None:
        output['txt'] = txt   

    if h_stat is not None:
        output['H_statistic'] = h_stat

    return output
        


def plot_distribution_summary(data, stats=None, distribution_type=None, title="Resumen de Distribución", x_label="Valor", y_label="Densidad", bins=30):
    """
    Crea una figura con:
      - Subplot superior izquierdo: Histograma con KDE.
      - Subplot superior derecho: Scatterplot (índice vs. valor).
      - Subplot inferior izquierdo: Boxplot.
      - Subplot inferior derecho: Tabla de estadísticos muestrales y la información de 'distribution_type'.

    Parámetros:
      data: array-like
          Los datos a visualizar.
      stats: dict, opcional
          Diccionario con estadísticos muestrales. Si no se provee, se calculan valores básicos.
      distribution_type: any, opcional
          Información sobre el tipo de distribución (por ejemplo, "Normal", "Poisson", etc.) que se incluirá en la tabla.
      title: str, opcional
          Título general de la figura.
      x_label: str, opcional
          Etiqueta del eje X para el histograma.
      y_label: str, opcional
          Etiqueta del eje Y para el histograma.
      bins: int, opcional
          Número de bins para el histograma.
    """
    # Calcular estadísticas básicas si no se provee un diccionario
    computed_stats = {
        "Count": len(data),
        "Mean": np.mean(data),
        "Median": np.median(data),
        "Std": np.std(data),
        "Min": np.min(data),
        "Max": np.max(data)
    }
    if stats is None:
        stats = computed_stats
    else:
        for key, value in computed_stats.items():
            if key not in stats:
                stats[key] = value

    if distribution_type is not None:
        stats["Distribution Type"] = distribution_type

    # Crear figura con 2 filas x 2 columnas
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    # Subplot superior izquierdo: Histograma con KDE
    ax1 = axes[0, 0]
    sns.histplot(data, bins=bins, kde=True, ax=ax1, color="skyblue", alpha=0.7)
    ax1.set_title("Histograma")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    # Subplot superior derecho: Scatterplot (índice vs. valor)
    ax2 = axes[0, 1]
    ax2.scatter(np.arange(len(data)), data, alpha=0.6, color="darkorange")
    ax2.set_title("Scatterplot")
    ax2.set_xlabel("Índice")
    ax2.set_ylabel("Valor")

    # Subplot inferior izquierdo: Boxplot
    ax3 = axes[1, 0]
    ax3.boxplot(data, vert=False, patch_artist=True,
                boxprops=dict(facecolor="lightgreen", color="green"),
                medianprops=dict(color="red"))
    ax3.set_title("Boxplot")

    # Subplot inferior derecho: Tabla de estadísticos
    ax4 = axes[1, 1]
    ax4.axis("tight")
    ax4.axis("off")
    table_data = [[str(key), f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)]
                  for key, value in stats.items()]
    table = ax4.table(cellText=table_data, colLabels=["Estadístico", "Valor"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    ax4.set_title("Estadísticos Muestrales")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
