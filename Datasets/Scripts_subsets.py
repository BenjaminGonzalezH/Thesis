"""
Script para reducción de dimensionalidad horizontal de los genes dentro
de cada dataset extraído de BARRA:CuRDa, esto considerando el criterio
HVG (high variable genes). En castellano, selección de aquellos genes que
presenten una mayor variabilidad dentro de las muestras extraídad.
==========================================
Enfoques:
Según el trabajo de Amezquita et. al. un enfoque simple para selección
de genes característicos en el dataset es usando HVG como criterio bajo
la premisa de que genes con alta varianza implican una señal de genuina
diferencia biológica entre los genes afectados.

No obstante, matematicamente existe la relación media-varianza, es decir,
que genes con una mayor expresión de media tienden a presentar una mayor
varianza (Basandose en una distribución de Poisson). Por ende, es necesario 
establecer 
 
Referencias:
Amezquita, R. A., Lun, A. T. L., Becht, E., Carey, V. J., Carpp, L. N., 
Geistlinger, L., Martini, F., Rue-Albrecht, K., Risso, D., Soneson, C., 
Waldron, L., Pagès, H., Smith, M. L., Huber, W., Morgan, M., Gottardo, R., 
& Hicks, S. C. (2020). Orchestrating single-cell analysis with Bioconductor. 
Nature Methods, 17(2), 137–145. https://doi.org/10.1038/s41592-019-0654-x
"""

##################################
# Imports
##################################
import numpy as np                                          # Aplicación de operaciones matematicas eficientes.
import pandas as pd                                         # Lectura de conjuntos de datos en csv.
from pathlib import Path                                    # Aplicar un formato generalizado de direcciones en el script.
import matplotlib.pyplot as plt                             # Funciones para gráficos.
import matplotlib.patches as mpatches


##################################
# Funciones secundarias
##################################
def load_dataset(filepath: Path):
    """
    Carga el CSV de BARRA:Curda.
    Estructura esperada:
      - Columna 'ID'    : identificador de muestra (P1N, P1T, ...)
      - Columna 'class' : etiqueta (Normal / Tumor)
      - Resto           : genes ENSG[ID]
      - Valores         : Expresión asociada al gen.
    """
    # Lectura de csv dentro de comprimidos en la carpeta especificada.
    df = pd.read_csv(filepath,compression='zip', header=0)
 
    # Extraer metadatos (Se evita mala lectura de clases por espacios en los
    # campos o formato de escritura).
    sample_ids = df["ID"].values
    y_labels   = df["class"].values
    y_clean = np.char.lower(np.char.strip(y_labels.astype(str)))
    y_binary   = (y_clean == "tumor").astype(int)   # 0=Normal, 1=Tumor
 
    # Extraer matriz de expresión (genes).
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    X = df[gene_cols].values.astype(np.float32)
 
    # Inferir grupos de paciente para GroupKFold
    # Convención: P1N → grupo 1, P1T → grupo 1, P2N → grupo 2, ...
    groups = np.array([
        int("".join(filter(str.isdigit, sid)))
        for sid in sample_ids
    ])
 
    print(f"\nDataset cargado: {filepath.parts[-1]}")
    print(f"  - Muestras  : {X.shape[0]}  ({sum(y_binary==0)} Normal, {sum(y_binary==1)} Tumor)")
    print(f"  - Genes     : {X.shape[1]:,}")
    print(f"  - rank máx. : {min(X.shape[0]-1, X.shape[1])}  (restricción matemática)")
    print(f"  - Grupos    : {np.unique(groups)}  (pacientes)\n")
 
    return X, y_binary, gene_cols, groups, sample_ids, df


def hvg_corrected(X: np.ndarray, gene_names: list, k: int = 1000,
                  n_bins: int = 20) -> np.ndarray:
    """
    Selección HVG modelando la relación media-varianza con la finalidad
    de que la elección del subset sea completamente guíada por genes
    con varianza superior a la esperada.
 
    Adaptado de la lógica de scran (Lun et al. 2016) para bulk RNA-seq:
      1. Calcular media y varianza por gen.
      2. Dividir genes en bins por expresión media.
      3. Dentro de cada bin, estimar la varianza "esperada" (mediana del bin).
      4. Calcular varianza residual = varianza_observada - varianza_esperada.
      5. Seleccionar top-K genes por varianza residual.
 
    Lun, A. T. L., McCarthy, D. J., & Marioni, J. C. (2016). A step-by-step 
    workflow for low-level analysis of single-cell RNA-seq data with Bioconductor. 
    F1000Research, 5, 2122. https://doi.org/10.12688/f1000research.9501.2
    """
    # Se realiza el cálculo de media y varianza de cada gen existente en el dataset.
    means     = np.mean(X, axis=0)
    variances = np.var(X, axis=0, ddof=1)
 
    # Dividir en bins de expresión media, es decir, se generan 20 percentiles
    # donde se organizan los genes con mayor expresión media.
    bin_edges  = np.percentile(means, np.linspace(0, 100, n_bins + 1))
    tech_var   = np.zeros_like(variances)

    # Por cada rango de valores de bines se realiza la identificación de
    # genes asociados a estos bines.
    for i in range(n_bins):
        # Rango proporcionado por bin_edges.
        lo, hi = bin_edges[i], bin_edges[i + 1]

        # Mascara aplicacda a cada gen perteneciente a ese bin.
        mask   = (means >= lo) & (means < hi) if i < n_bins - 1 \
                 else (means >= lo) & (means <= hi)
        
        # Si existen genes en ese bin, se optiene la mediana de
        # las varianzas (evita inestabilidad por el promedio).
        if mask.sum() > 0:
            tech_var[mask] = np.median(variances[mask])
 
    # La varianza biológica en este caso se considera la varianza
    # exedente a lo que por su propio bin, es considerado más de lo
    # esperado.
    bio_var = variances - tech_var
 
    # Selección: top-K por varianza biológica (solo positiva) y se genera un
    # subset de genes con alta varianza para lo esperado.
    min_bio_var  = 0.0
    candidates   = np.where(bio_var > min_bio_var)[0]
    ranked       = candidates[np.argsort(bio_var[candidates])[::-1]]
    hvg_idx      = np.sort(ranked[:k])
 
    print(f"\n[B] HVG corregido: seleccionados {k} genes por varianza biológica residual")
    print(f"    Genes con varianza residual > 0 : {len(candidates):,} / {X.shape[1]:,}")
    print(f"    Varianza biológica (top {k}): "
          f"{bio_var[hvg_idx].min():.3f} – {bio_var[hvg_idx].max():.3f}")
 
    return hvg_idx, means, variances, tech_var, bio_var


def save_reduced_dataset(df_original: pd.DataFrame,
                         hvg_idx: np.ndarray,
                         gene_names: list,
                         output_dir: Path,
                         source_name: str,
                         k: int) -> Path:
    """
    Construye y guarda el dataset reducido conservando las columnas
    de metadatos (ID, class) y reteniendo solo los K genes seleccionados;
    considerando el almacenamiento dentro de un directorio propio.
    """
    # Se asegura que el directorio exista y se mantienen las columnas
    # de metadatos.
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_cols = ["ID", "class"]
 
    # Se obtienen columnas de los genes seleccionados.
    selected_genes = [gene_names[i] for i in hvg_idx]
 
    # Construir DataFrame reducido: metadatos + genes seleccionados conservando
    # valores de expresión dentro de las muestras existentes.
    df_reduced = df_original[meta_cols + selected_genes].copy()
 
    # Nombre del archivo de salida <dataset>_HVG_top<K>.csv es en donde
    # se almacena el dataset reducido.
    stem = source_name.replace(".csv", "").replace(".zip", "")
    out_path = output_dir / f"{stem}_HVG_top{k}.csv"
    df_reduced.to_csv(out_path, index=False)
 
    print(f"[SAVE] Dataset reducido guardado en:")
    print(f"       {out_path}")
    print(f"       Dimensión: {df_reduced.shape[0]} muestras × {len(selected_genes)} genes\n")
 
    return out_path


def plot_bins_hvg(means: np.ndarray,
                  variances: np.ndarray,
                  tech_var: np.ndarray,
                  bio_var: np.ndarray,
                  hvg_idx: np.ndarray,
                  bin_edges: np.ndarray,
                  output_dir: Path,
                  source_name: str,
                  k: int) -> Path:
    """
    Genera un gráfico de dispersión media-varianza que muestra:
      - Todos los genes (puntos grises).
      - La mediana de varianza técnica por bin (línea horizontal roja
        dentro de cada franja coloreada).
      - Los K genes HVG seleccionados resaltados en púrpura.
 
    El eje Y muestra varianza observada. Los genes HVG son aquellos
    cuya varianza supera la mediana de su bin (varianza biológica > 0).
 
    Parámetros
    ----------
    means, variances : arrays de media y varianza por gen.
    tech_var         : varianza técnica estimada por bin para cada gen.
    bio_var          : varianza biológica residual por gen.
    hvg_idx          : índices de los genes HVG seleccionados.
    bin_edges        : bordes de los bins (n_bins + 1 valores).
    output_dir       : carpeta de destino.
    source_name      : nombre base del dataset.
    k                : número de genes seleccionados.
 
    Retorna
    -------
    Path del archivo PNG guardado.
    """
    # Se asegura existencia del directorio para almacenar los resultados.
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Se establecen los rangos de bines previamente calculados, asimismo,
    # mediante la discriminación de genes HVG y no HVG se determinaran
    # los colores de los puntos.
    n_bins    = len(bin_edges) - 1
    hvg_set   = set(hvg_idx)
    non_hvg   = np.array([i for i in range(len(means)) if i not in hvg_set])

    # Creación de marco de los gráficos (2 en total).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    fig.suptitle(
        f"Selección HVG por bins de expresión media  ·  {source_name}  ·  top {k} genes",
        fontsize=12, fontweight="bold", y=1.01
    )
 
    # ── Panel izquierdo: dispersión media-varianza ───────────────────
    ax = axes[0]
 
    # Colores alternos para los bins (fondo)
    bin_colors = ["#EEEDFE", "#E1F5EE"]
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        ax.axvspan(lo, hi, alpha=0.35,
                   color=bin_colors[i % 2], zorder=0)
 
        # Mediana de varianza técnica del bin (línea horizontal roja)
        mask = (means >= lo) & (means < hi) if i < n_bins - 1 \
               else (means >= lo) & (means <= hi)
        if mask.sum() > 0:
            med = np.median(variances[mask])
            ax.hlines(med, lo, hi,
                      colors="#D85A30", linewidths=1.4,
                      linestyles="--", zorder=2)
 
    # Genes no seleccionados
    ax.scatter(means[non_hvg], variances[non_hvg],
               s=2, alpha=0.25, color="#B4B2A9",
               rasterized=True, zorder=1, label="Resto de genes")
 
    # Genes HVG seleccionados
    ax.scatter(means[hvg_idx], variances[hvg_idx],
               s=14, alpha=0.85, color="#534AB7",
               zorder=3, label=f"HVG seleccionados (n={k})")
 
    ax.set_xlabel("Expresión media (VST)", fontsize=10)
    ax.set_ylabel("Varianza observada", fontsize=10)
    ax.set_title("Dispersión media-varianza\n(línea roja = mediana técnica del bin)",
                 fontsize=10)
 
    legend_elements = [
        mpatches.Patch(color="#B4B2A9", alpha=0.6, label="Genes no seleccionados"),
        mpatches.Patch(color="#534AB7",             label=f"HVG top {k}"),
        plt.Line2D([0], [0], color="#D85A30", lw=1.4, ls="--",
                   label="Mediana técnica por bin"),
        mpatches.Patch(color="#EEEDFE", alpha=0.8, label="Bins alternos"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    ax.tick_params(labelsize=9)
 
    # ── Panel derecho: varianza biológica residual ───────────────────
    ax2 = axes[1]
 
    bio_non = bio_var[non_hvg]
    bio_hvg = bio_var[hvg_idx]
 
    ax2.scatter(means[non_hvg], bio_non,
                s=2, alpha=0.25, color="#B4B2A9",
                rasterized=True, zorder=1, label="Varianza residual no seleccionada")
    ax2.scatter(means[hvg_idx], bio_hvg,
                s=14, alpha=0.85, color="#534AB7",
                zorder=3, label=f"HVG top {k}")
 
    # Umbral en 0: genes por encima son candidatos HVG
    ax2.axhline(0, color="#D85A30", linewidth=1.2,
                linestyle="--", zorder=2, label="Umbral residual = 0")
 
    # Franja de bins (fondo)
    for i in range(n_bins):
        ax2.axvspan(bin_edges[i], bin_edges[i + 1],
                    alpha=0.2, color=bin_colors[i % 2], zorder=0)
 
    ax2.set_xlabel("Expresión media (VST)", fontsize=10)
    ax2.set_ylabel("Varianza biológica residual\n(observada − mediana del bin)", fontsize=10)
    ax2.set_title("Varianza residual por gen\n(HVG = genes sobre el umbral 0)",
                  fontsize=10)
 
    legend2 = [
        mpatches.Patch(color="#B4B2A9", alpha=0.6, label="Residual no seleccionado"),
        mpatches.Patch(color="#534AB7",             label=f"HVG top {k}"),
        plt.Line2D([0], [0], color="#D85A30", lw=1.2, ls="--", label="Umbral = 0"),
    ]
    ax2.legend(handles=legend2, fontsize=8, loc="upper left")
    ax2.tick_params(labelsize=9)
 
    plt.tight_layout()
 
    stem     = source_name.replace(".csv", "").replace(".zip", "")
    out_path = output_dir / f"{stem}_HVG_bins_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
    print(f"[PLOT] Gráfico guardado en:")
    print(f"       {out_path}\n")
 
    return out_path


##################################
# Ejecución principal
##################################
if __name__ == "__main__":
    # ── Configuración ────────────────────────────────────────────────
    source_file = Path(__file__).resolve().parent / "Parameterization" / "GSE63511.csv.zip"
    output_dir  = Path(__file__).resolve().parent / "Reduced Datasets"
    cant_genes  = 500
    n_bins      = 20
 
    # ── Carga ────────────────────────────────────────────────────────
    X, y, gene_names, groups, sample_ids, df_original = load_dataset(source_file)
 
    # ── Selección HVG ────────────────────────────────────────────────
    hvg_idx, means, variances, tech_var, bio_var = hvg_corrected(
        X, gene_names, k=cant_genes, n_bins=n_bins
    )
 
    # Recalcular bin_edges para pasarlos al gráfico
    bin_edges = np.percentile(means, np.linspace(0, 100, n_bins + 1))
 
    # ── Función 1: guardar dataset reducido ──────────────────────────
    save_reduced_dataset(
        df_original = df_original,
        hvg_idx     = hvg_idx,
        gene_names  = gene_names,
        output_dir  = output_dir,
        source_name = source_file.name,
        k           = cant_genes
    )
 
    # ── Función 2: gráfico de bins y HVG ────────────────────────────
    plot_bins_hvg(
        means       = means,
        variances   = variances,
        tech_var    = tech_var,
        bio_var     = bio_var,
        hvg_idx     = hvg_idx,
        bin_edges   = bin_edges,
        output_dir  = output_dir,
        source_name = source_file.name,
        k           = cant_genes
    )
 
    print("Proceso completado.")

