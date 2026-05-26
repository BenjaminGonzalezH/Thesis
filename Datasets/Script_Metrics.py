"""
Script para la construcción de matrices de distancia asociadas a los genes, 
considerando el cambio de identificadores de genes y un pre-estudio de los
datos para verificar condiciones de qué métrica aplicar para la matriz de 
distancia de expresión génica.
==========================================
Enfoques:
  1. EDA visual       → boxplot por muestra, normalidad (Shapiro-Wilk), varianza por gen
  2. Mapeo de genes   → Ensembl ID → símbolo génico (mygene)
  3. Distancia        → Spearman (d = (1 − ρ) / 2) como métrica principal
  4. Heatmap          → visualización de la matriz de distancia gen × gen

Referencia del dataset:
  GSE63511 - Top 500 genes de alta varianza (HVGs)
"""

##################################
# Imports
##################################
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import shapiro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
import time
import requests
from io import StringIO
from typing import List, Sequence, Set, Tuple, Union, Literal
from dataclasses import dataclass

# go3 es opcional: solo se requiere para similitud semántica GO
try:
    import go3
    GO3_AVAILABLE = True
except ImportError:
    GO3_AVAILABLE = False
    print("⚠ go3 no está instalado. Las funciones de similitud GO no estarán disponibles.")

import mygene

warnings.filterwarnings("ignore")


##################################
# CONFIGURACIÓN PRINCIPAL
##################################
# ─── Editar estas rutas para adaptar el script a cualquier dataset ───────────
SOURCE_FILE = Path(__file__).resolve().parent / "Reduced Datasets" / "GSE63511_HVG_top500.csv"
OUTPUT_DIR  = Path(__file__).resolve().parent / "Metrics Sources" / "GSE63511"
SPECIES     = "human"       # especie para mygene (human, mouse, rat, ...)
SPECIES_ID  = 9606          # NCBI taxonomy ID para STRING API (human=9606)
# ─────────────────────────────────────────────────────────────────────────────


##################################
# CONSTANTES
##################################
C_TUMOR   = "#E63946"
C_NORMAL  = "#457B9D"
C_BG      = "#F8F9FA"
C_TITLE   = "#1d3557"
CMAP_HEAT = LinearSegmentedColormap.from_list(
    "spearman", ["#1d3557", "#457b9d", "#f1faee", "#e9c46a", "#E63946"]
)

Ontology          = Literal["BP", "MF", "CC"]
Groupwise         = Literal["bma", "max", "avg", "hausdorff", "simgic"]
SimilarityMeasure = Literal["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
DistanceTransform = Literal["auto", "one_minus", "max_minus", "reciprocal"]

STRING_API_URL = "https://version-12-0.string-db.org/api"
OUTPUT_FORMAT  = "tsv"


@dataclass(frozen=True)
class GeneSimilarityOptions:
    ontology:        Ontology         = "BP"
    measure:         SimilarityMeasure = "wang"
    groupwise:       Groupwise         = "bma"
    distance_method: DistanceTransform = "auto"
    load_go_terms:   bool              = True
    num_threads_go3: int               = 0   # 0 = auto


##################################
# Funciones secundarias
##################################

# -------------------------------------------------- Carga de datos
def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Lee la matriz de expresión desde un CSV.

    Returns
    -------
    df     : DataFrame completo.
    labels : Serie con etiquetas de clase (Tumor / Normal).
    expr   : DataFrame con valores de expresión (genes como columnas).
    """
    df = pd.read_csv(path)

    # Detecta columnas de metadatos de forma flexible
    id_col    = [c for c in df.columns if c.upper() in ("ID", "SAMPLE", "SAMPLEID")]
    class_col = [c for c in df.columns if c.lower() in ("class", "group", "label", "condition")]

    drop   = id_col + class_col
    expr   = df.drop(columns=drop)
    labels = df[class_col[0]].str.strip() if class_col else pd.Series(["unknown"] * len(df))
    return df, labels, expr


# -------------------------------------------------- Mapeo de genes
def map_ensembl_to_symbol(ensembl_ids: List[str], species: str = "human") -> pd.DataFrame:
    """
    Convierte una lista de Ensembl Gene IDs a símbolos génicos usando mygene.

    Parameters
    ----------
    ensembl_ids : list of str
        Lista de IDs tipo ENSG00000XXXXXX.
    species : str
        Especie (human, mouse, rat, …).

    Returns
    -------
    pd.DataFrame con columnas: ensembl_id, symbol
    """
    mg      = mygene.MyGeneInfo()
    # getgenes realiza la consulta en lote (una sola llamada a la API)
    results = mg.getgenes(ensembl_ids, fields="symbol", species=species)

    rows = []
    for r in results:
        rows.append({
            "ensembl_id": r.get("query", ""),
            "symbol":     r.get("symbol", r.get("query", "")),   # fallback al ID si no hay símbolo
        })
    return pd.DataFrame(rows)


# -------------------------------------------------- Distancias
def dist_spearman(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Distancia de Spearman entre genes:  d(g_i, g_j) = (1 − ρ_s) / 2

    Cada gen se representa como un vector de expresión a través de las
    n_muestras muestras. La correlación de Spearman mide si sus perfiles
    covarían monótonamente entre muestras.

    Entrada : expr → shape (n_muestras × n_genes)
    Salida  : mat  → shape (n_genes × n_genes),  rango [0, 1]

    Referencias
    -----------
    Hou et al. (2022). BMC Bioinformatics, 23(1), 81.
    Mutwil et al. (2018). Scientific Reports, 8, 10695.
    """
    # Transponer: ahora cada fila es un gen y cada columna una muestra,
    # de modo que spearmanr compara perfiles de expresión entre genes.
    genes = expr.columns.tolist()
    vals  = expr.values.T          # shape: (n_genes, n_muestras)
    n     = len(genes)
    mat   = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            rho, _        = spearmanr(vals[i], vals[j])
            mat[i, j]     = (1.0 - rho) / 2
            mat[j, i]     = mat[i, j]

    return pd.DataFrame(mat, index=genes, columns=genes)


# -------------------------------------------------- Similitud GO (go3)
def compute_gene_similarity_matrix_go3(
    genes: Sequence[str],
    *,
    obo_path: Union[str, Path],
    gaf_path: Union[str, Path],
    go3_opts: GeneSimilarityOptions = GeneSimilarityOptions(),
) -> Tuple[List[str], np.ndarray]:
    """
    Calcula la matriz de similitud semántica génica usando go3.

    Requiere que go3 esté instalado (GO3_AVAILABLE = True).

    Parameters
    ----------
    genes    : lista de identificadores de gen.
    obo_path : ruta al archivo .obo de Gene Ontology.
    gaf_path : ruta al archivo GAF de anotaciones.
    go3_opts : opciones de configuración de go3.

    Returns
    -------
    (ordered_genes, similarity_matrix)
    """
    if not GO3_AVAILABLE:
        raise RuntimeError("go3 no está instalado. Instálalo con: pip install go3")

    if len(genes) == 0:
        raise ValueError("genes debe ser una lista no vacía.")
    if go3_opts.ontology not in ("BP", "MF", "CC"):
        raise ValueError("ontology debe ser 'BP', 'MF' o 'CC'.")

    go3.set_num_threads(int(go3_opts.num_threads_go3))

    if go3_opts.load_go_terms:
        go3.load_go_terms(str(obo_path))

    annotations = go3.load_gaf(str(gaf_path))
    counter     = go3.build_term_counter(annotations)

    ordered, dist = go3.gene_distance_matrix(
        genes,
        ontology          = go3_opts.ontology,
        similarity        = go3_opts.measure,
        groupwise         = go3_opts.groupwise,
        counter           = counter,
        distance_transform= go3_opts.distance_method,
    )

    sim = 1.0 - np.array(dist, dtype=np.float64)
    return list(ordered), sim


# -------------------------------------------------- STRING PPI
def map_genes_to_string_ids(
    genes: List[str],
    species: int = 9606,
    caller_identity: str = "gclusters_characterization",
) -> pd.DataFrame:
    """
    Mapea símbolos génicos a identificadores de proteína en STRING.

    Parameters
    ----------
    genes           : lista de símbolos génicos.
    species         : NCBI taxonomy ID (human = 9606).
    caller_identity : identificador del script para la API de STRING.

    Returns
    -------
    DataFrame con columnas queryItem, stringId, preferredName.
    """
    url    = f"{STRING_API_URL}/{OUTPUT_FORMAT}/get_string_ids"
    params = {
        "identifiers":   "\r".join(genes),
        "species":       species,
        "limit":         1,
        "echo_query":    1,
        "caller_identity": caller_identity,
    }
    r = requests.post(url, data=params, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), sep="\t") if r.text.strip() else pd.DataFrame()


def get_string_network(
    string_ids: List[str],
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    caller_identity: str = "gclusters_characterization",
) -> pd.DataFrame:
    """
    Recupera interacciones de STRING para una lista de IDs de proteína.

    Parameters
    ----------
    string_ids     : IDs de proteína STRING.
    species        : NCBI taxonomy ID.
    required_score : umbral de confianza STRING (0–1000).
                     150=bajo, 400=medio, 700=alto, 900=más alto.
    network_type   : "functional" o "physical".

    Returns
    -------
    DataFrame con la tabla de interacciones de STRING.
    """
    url    = f"{STRING_API_URL}/{OUTPUT_FORMAT}/network"
    params = {
        "identifiers":   "\r".join(string_ids),
        "species":       species,
        "required_score": required_score,
        "network_type":  network_type,
        "caller_identity": caller_identity,
    }
    r = requests.post(url, data=params, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), sep="\t") if r.text.strip() else pd.DataFrame()


def build_string_ppi_similarity_matrix(
    genes: List[str],
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    missing_value: float = 0.0,
    caller_identity: str = "gclusters_characterization",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construye una matriz de similitud gen–gen a partir de scores de confianza STRING.

    Parameters
    ----------
    genes          : símbolos génicos de entrada.
    species        : NCBI taxonomy ID.
    required_score : umbral mínimo de confianza STRING.
    network_type   : "functional" o "physical".
    missing_value  : valor para pares sin interacción en STRING.

    Returns
    -------
    (similarity_matrix, mapping_df, network_df)
    """
    genes = list(dict.fromkeys(genes))   # elimina duplicados preservando orden

    mapping_df = map_genes_to_string_ids(genes, species, caller_identity)
    if mapping_df.empty:
        raise ValueError("Ningún gen pudo mapearse a STRING.")

    mapping_df     = mapping_df.drop_duplicates(subset=["queryItem"], keep="first")
    q_to_str       = dict(zip(mapping_df["queryItem"], mapping_df["stringId"]))
    str_to_q       = dict(zip(mapping_df["stringId"],  mapping_df["queryItem"]))
    mapped_genes   = [g for g in genes if g in q_to_str]
    string_ids     = [q_to_str[g] for g in mapped_genes]

    if len(string_ids) < 2:
        raise ValueError("Se necesitan al menos dos genes mapeados para construir la matriz.")

    time.sleep(1)   # respeto al rate-limit de STRING

    network_df = get_string_network(string_ids, species, required_score,
                                    network_type, caller_identity)

    sim = pd.DataFrame(missing_value, index=mapped_genes,
                       columns=mapped_genes, dtype=float)
    np.fill_diagonal(sim.values, 1.0)

    if not network_df.empty:
        for _, row in network_df.iterrows():
            ga = str_to_q.get(row["stringId_A"])
            gb = str_to_q.get(row["stringId_B"])
            if ga and gb:
                score = float(row["score"])
                sim.loc[ga, gb] = score
                sim.loc[gb, ga] = score

    return sim, mapping_df, network_df


# -------------------------------------------------- Visualizaciones
def plot_eda(expr: pd.DataFrame, labels: pd.Series, out_path: str, title: str) -> pd.DataFrame:
    """
    Genera tres gráficos de exploración de los datos:
      A) Boxplot de expresión por muestra
      B) Histograma de p-valores Shapiro-Wilk (normalidad por gen)
      C) Histograma de varianza por gen

    Returns
    -------
    norm_df : DataFrame con p-valor, flag de normalidad y varianza por gen.
    """
    pvals    = np.array([shapiro(expr[col])[1] for col in expr.columns])
    var_gen  = expr.var(axis=0)
    n_norm   = (pvals > 0.05).sum()
    pct_norm = n_norm / len(pvals) * 100

    sample_colors = [C_TUMOR if l == "Tumor" else C_NORMAL for l in labels]
    short_ids     = [f"{'T' if l=='Tumor' else 'N'}{i+1}" for i, l in enumerate(labels)]

    fig = plt.figure(figsize=(18, 5), facecolor=C_BG)
    fig.suptitle(f"Análisis Exploratorio – {title}",
                 fontsize=14, fontweight="bold", color=C_TITLE, y=1.02)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── A · Boxplot por muestra ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bp  = ax1.boxplot(
        expr.T.values, patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], sample_colors):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    for flier, color in zip(bp["fliers"], sample_colors):
        flier.set_markerfacecolor(color); flier.set_markeredgecolor(color)

    ax1.set_xticks(range(1, len(labels) + 1))
    ax1.set_xticklabels(short_ids, fontsize=8, rotation=45, ha="right")
    ax1.set_ylabel("log₂(expresión)", fontsize=9)
    ax1.set_title("Distribución de expresión\npor muestra", fontweight="bold", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(handles=[
        mpatches.Patch(facecolor=C_TUMOR,  label="Tumor"),
        mpatches.Patch(facecolor=C_NORMAL, label="Normal"),
    ], fontsize=9, loc="upper right")
    med_global = np.median(expr.values)
    ax1.axhline(med_global, color="#264653", lw=1, linestyle=":",
                alpha=0.6, label=f"Mediana global={med_global:.2f}")

    # ── B · Normalidad Shapiro-Wilk ──────────────────────────────────────────
    ax2   = fig.add_subplot(gs[1])
    edges = np.linspace(0, 1, 31)
    bins_c, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    c_norm,    _ = np.histogram(pvals[pvals > 0.05],  bins=edges)
    c_notnorm, _ = np.histogram(pvals[pvals <= 0.05], bins=edges)

    ax2.bar(bins_c, c_norm + c_notnorm, width=width * 0.9,
            color="#2a9d8f", alpha=0.85, edgecolor="white",
            linewidth=0.3, label="Normal (p>0.05)")
    ax2.bar(bins_c, c_notnorm, width=width * 0.9,
            color="#e76f51", alpha=0.85, edgecolor="white",
            linewidth=0.3, label="No normal (p≤0.05)")
    ax2.axvline(0.05, color=C_TUMOR, lw=2, linestyle="--", label="α = 0.05")
    ax2.text(0.55, 0.88, f"{pct_norm:.1f}% normales\n({n_norm}/{len(pvals)} genes)",
             transform=ax2.transAxes, fontsize=9, color="#2a9d8f", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax2.set_xlabel("p-valor", fontsize=9)
    ax2.set_ylabel("Nº genes", fontsize=9)
    ax2.set_title("P-valores Shapiro-Wilk\n(normalidad por gen)", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── C · Varianza por gen ──────────────────────────────────────────────────
    ax3      = fig.add_subplot(gs[2])
    high_var = (var_gen > 10).sum()
    ax3.hist(var_gen, bins=40, color="#f4a261", alpha=0.85,
             edgecolor="white", linewidth=0.3)
    ax3.axvline(var_gen.mean(),   color=C_TUMOR,   lw=2,   linestyle="--",
                label=f"Media = {var_gen.mean():.2f}")
    ax3.axvline(var_gen.median(), color="#264653", lw=1.5, linestyle=":",
                label=f"Mediana = {var_gen.median():.2f}")
    ax3.axvspan(10, var_gen.max() + 0.5, alpha=0.12, color=C_TUMOR,
                label=f"Var > 10 ({high_var} genes)")
    ax3.set_xlabel("Varianza", fontsize=9)
    ax3.set_ylabel("Nº genes", fontsize=9)
    ax3.set_title("Distribución de varianza\npor gen", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  → EDA guardado: {out_path}")

    return pd.DataFrame({
        "gen":       expr.columns,
        "p_shapiro": pvals,
        "normal":    pvals > 0.05,
        "varianza":  var_gen.values,
    })


def plot_heatmap(dist_mat: pd.DataFrame, out_path: str, title: str = "") -> None:
    """
    Heatmap de la matriz de distancia Spearman gen × gen con colorbar lateral.

    Parameters
    ----------
    dist_mat : DataFrame cuadrado con las distancias gen × gen.
    out_path : ruta de salida de la imagen.
    title    : subtítulo descriptivo del dataset (se añade al título del gráfico).
    """
    vals  = dist_mat.values
    n     = len(dist_mat)
    vmax  = vals.max()               # se deriva de los datos, no hardcodeado

    fig, (ax_heat, ax_cb) = plt.subplots(
        1, 2, figsize=(10, 9), facecolor=C_BG,
        gridspec_kw={"width_ratios": [1, 0.04], "wspace": 0.04},
    )
    suptitle = f"Matriz de Distancia Spearman  (d = (1 − ρ) / 2)  |  Gen × Gen"
    if title:
        suptitle += f"\n{title}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=C_TITLE, y=1.01)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    im = ax_heat.imshow(vals, cmap=CMAP_HEAT, aspect="auto",
                        vmin=0, vmax=vmax, interpolation="nearest")
    ax_heat.set_xticks([]); ax_heat.set_yticks([])
    ax_heat.set_xlabel(f"Genes (n={n})", fontsize=10)
    ax_heat.set_ylabel(f"Genes (n={n})", fontsize=10)
    ax_heat.spines[:].set_visible(False)

    # ── Colorbar lateral ──────────────────────────────────────────────────────
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label("Distancia Spearman", fontsize=9, labelpad=8)
    cb.ax.tick_params(labelsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  → Heatmap guardado: {out_path}")


##################################
# Pipeline principal
##################################
##################################
# Pipeline principal
##################################
def run(
    source_file:     Path = SOURCE_FILE,
    output_dir:      Path = OUTPUT_DIR,
    species:         str  = SPECIES,
    obo_path:        Path | None = None,
    gaf_path:        Path | None = None,
    go3_opts:        GeneSimilarityOptions = GeneSimilarityOptions(),
    string_score:    int  = 400,
) -> None:
    """
    Ejecuta el pipeline completo:
      1. Carga del dataset
      2. Mapeo Ensembl ID → símbolo génico
      3. EDA (boxplot por muestra, normalidad Shapiro-Wilk, varianza por gen)
      4. Distancia de expresión Spearman gen × gen
      5. Similitud PPI STRING gen × gen       (opcional, requiere red STRING)
      6. Similitud semántica GO3 gen × gen    (opcional, requiere go3 + archivos OBO/GAF)
 
    Parameters
    ----------
    source_file  : ruta al CSV de expresión.
    output_dir   : directorio donde se guardan los resultados.
    species      : especie para mygene (human, mouse, rat, …).
    obo_path     : ruta al archivo .obo de Gene Ontology (paso 6).
    gaf_path     : ruta al archivo GAF de anotaciones (paso 6).
    go3_opts     : configuración de similitud semántica GO3 (paso 6).
    string_score : umbral mínimo de confianza STRING 0-1000 (paso 5).
    """
    # ── 0. Preparar directorio de salida ──────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = source_file.stem      # nombre del archivo sin extensión
 
    print(f"📂 Dataset : {source_file.name}")
    print(f"📁 Salida  : {output_dir}")
 
    # ── 1. Carga ──────────────────────────────────────────────────────────────
    print("\n[1/6] Cargando datos...")
    df, labels, expr = load_dataset(source_file)
    print(f"      {expr.shape[0]} muestras × {expr.shape[1]} genes")
    print(f"      Clases: {labels.value_counts().to_dict()}")
 
    # ── 2. Mapeo Ensembl ID → símbolo génico ──────────────────────────────────
    print("\n[2/6] Mapeando Ensembl IDs → símbolos génicos...")
    mapping_df   = map_ensembl_to_symbol(expr.columns.tolist(), species=species)
    mapping_path = output_dir / f"{dataset_name}_gene_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    mapped = mapping_df[mapping_df["symbol"] != mapping_df["ensembl_id"]]
    print(f"      {len(mapped)}/{len(mapping_df)} genes mapeados correctamente")
    print(f"      Mapeo guardado: {mapping_path}")
 
    # Obtener lista de símbolos para los pasos STRING y GO3
    gene_symbols = mapping_df["symbol"].tolist()
 
    # ── 3. EDA ────────────────────────────────────────────────────────────────
    print("\n[3/6] Generando EDA...")
    norm_df   = plot_eda(
        expr, labels,
        out_path = str(output_dir / f"{dataset_name}_eda.png"),
        title    = dataset_name,
    )
    norm_path = output_dir / f"{dataset_name}_normalidad_por_gen.csv"
    norm_df.to_csv(norm_path, index=False)
    print(f"      Genes normales (p>0.05): {norm_df['normal'].sum()} / {len(norm_df)}"
          f" ({norm_df['normal'].mean()*100:.1f}%)")
    print(f"      Tabla guardada: {norm_path}")
 
    # ── 4. Distancia Spearman gen × gen ───────────────────────────────────────
    print(f"\n[4/6] Calculando distancia de Spearman ({expr.shape[1]}×{expr.shape[1]})...")
    print(f"      Pares a calcular: {expr.shape[1] * (expr.shape[1]-1) // 2:,}")
    dist_mat  = dist_spearman(expr)
    dist_path = output_dir / f"{dataset_name}_dist_spearman.csv"
    dist_mat.to_csv(dist_path)
    print(f"      Matriz guardada: {dist_path}")
    plot_heatmap(
        dist_mat,
        out_path = str(output_dir / f"{dataset_name}_spearman_heatmap.png"),
        title    = dataset_name,
    )
 
    # ── 5. Similitud PPI STRING gen × gen ─────────────────────────────────────
    print("\n[5/6] Calculando similitud PPI STRING...")
    try:
        sim_string, mapping_string, network_df = build_string_ppi_similarity_matrix(
            genes          = gene_symbols,
            species        = SPECIES_ID,
            required_score = string_score,
        )
        string_path = output_dir / f"{dataset_name}_sim_string.csv"
        sim_string.to_csv(string_path)
        plot_heatmap(
            sim_string,
            out_path = str(output_dir / f"{dataset_name}_string_heatmap.png"),
            title    = dataset_name,
            )
        print(f"      Matriz STRING guardada: {string_path}")
    except Exception as e:
        print(f"      ⚠ STRING omitido: {e}")
 
    # ── 6. Similitud semántica GO3 gen × gen ──────────────────────────────────
    print("\n[6/6] Calculando similitud semántica GO3...")
    if not GO3_AVAILABLE:
        print("      ⚠ GO3 omitido: módulo no instalado.")
    elif obo_path is None or gaf_path is None:
        print("      ⚠ GO3 omitido: proporciona obo_path y gaf_path en run().")
    else:
        try:
            gene_symbols = pd.read_csv(r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\Metrics Sources\GSE63511_HVG_top500_gene_mapping_edit.csv")
            gene_symbols = list(gene_symbols["symbol"])
            ordered_genes, sim_go3_arr = compute_gene_similarity_matrix_go3(
                genes    = gene_symbols,
                obo_path = obo_path,
                gaf_path = gaf_path,
                go3_opts = go3_opts,
            )
            sim_go3      = pd.DataFrame(sim_go3_arr,
                                        index=ordered_genes, columns=ordered_genes)
            go3_path     = output_dir / f"{dataset_name}_sim_go3.csv"
            sim_go3.to_csv(go3_path)
            plot_heatmap(
            sim_go3,
            out_path = str(output_dir / f"{dataset_name}_wang_heatmap.png"),
            title    = dataset_name,
            )
            print(f"      Matriz GO3 guardada: {go3_path}")
        except Exception as e:
            print(f"      ⚠ GO3 omitido: {e}")
 
    print("\n✅ Pipeline completado.")


# ─────────────────────────────────────────────────────────────────────────────
run(obo_path=r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\Bio_Info Resources\go.obo",
    gaf_path=r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\Bio_Info Resources\goa_human.gaf")