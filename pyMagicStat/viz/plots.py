import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, Optional

def plot_distribution_summary(
    data: np.ndarray,
    stats: Optional[Dict[str, Any]] = None,
    distribution_type: Optional[Any] = None,
    title: str = "Resumen de Distribución",
    x_label: str = "Valor",
    y_label: str = "Densidad",
    bins: int = 30
) -> None:
    """
    Crea una figura con un resumen visual de la distribución de los datos.
    
    Genera 4 subplots:
      - Superior izquierdo: Histograma con estimación de densidad del kernel (KDE).
      - Superior derecho: Gráfico de dispersión (Scatterplot) de índice vs. valor.
      - Inferior izquierdo: Diagrama de caja (Boxplot).
      - Inferior derecho: Tabla con estadísticos muestrales descriptivos.

    Parameters
    ----------
    data : np.ndarray
        Los datos numéricos a visualizar.
    stats : Dict[str, Any], optional
        Diccionario precalculado con estadísticos muestrales. Si es None, 
        se calculan internamente valores básicos (Media, Mediana, etc.).
    distribution_type : Any, optional
        Información sobre el tipo de distribución (ej. "Normal", "Poisson") a mostrar.
    title : str, default="Resumen de Distribución"
        Título principal de la figura.
    x_label : str, default="Valor"
        Etiqueta del eje X para el histograma y boxplot.
    y_label : str, default="Densidad"
        Etiqueta del eje Y para el histograma.
    bins : int, default=30
        Número de contenedores (bins) para el histograma.

    Returns
    -------
    None
        La función renderiza y muestra el gráfico directamente mediante `plt.show()`.
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
