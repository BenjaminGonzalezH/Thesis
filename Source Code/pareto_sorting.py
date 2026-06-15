"""
pareto_sorting.py
------------------
Ordenamiento por dominancia de Pareto (non-dominated sorting) y crowding
distance aplicados a una POBLACIÓN de soluciones, manteniendo el esquema
de NSGA-II usado en MOC-GaPBK (Parraga-Alava et al., 2018):

    Rv = NON-DOMINATEDSORTING-CROWDINGDISTANCE(Rv)

Es decir, dada una población de soluciones evaluadas en (XBEB, XBBB),
este módulo:

    1. Las clasifica en frentes de Pareto F1, F2, ... (non_dominated_sort).
    2. Calcula la crowding distance de cada solución dentro de su frente.
    3. Asigna rank (nivel de no-dominancia) + crowding distance a cada
       individuo de la población (non_dominated_sorting_crowding_distance),
       insumo estándar para selección por torneo binario en NSGA-II.
    4. Permite visualizar los frentes de Pareto en un plano 2D (XBEB vs XBBB).

Este módulo es independiente de las matrices DEB/DBB: opera directamente
sobre pares de objetivos (XBEB, XBBB) ya calculados, o puede calcularlos
internamente para una población completa de medoides.
"""

##################################
# Imports
##################################
import numpy as np
import pandas as pd
from pathlib import Path

from solution_definition import build_clusters, xie_beni_eb

# ══════════════════════════════════════════════════════════════════════════════
# 1. EVALUACIÓN DE OBJETIVOS (XBEB, XBBB)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_objectives(
    medoids: np.ndarray,
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
) -> tuple[float, float]:
    """
    Evalúa los dos objetivos de minimización (XBEB, XBBB) para una solución.

    Cada objetivo usa su propia asignación de clusters: XBEB se calcula
    con clusters formados según DEB, XBBB con clusters formados según DBB,
    replicando la Tabla 1 del paper.

    Parámetros
    ----------
    medoids    : np.ndarray — arreglo 1-D de k índices de medoides (0-based).
    deb_matrix : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    dbb_matrix : np.ndarray — matriz (n×n) de distancia biológica (DBB).

    Retorna
    -------
    tuple[float, float] — (XBEB, XBBB). Ambos a minimizar.

    Ejemplo
    -------
    >>> xbeb, xbbb = evaluate_objectives(medoids, deb_matrix, dbb_matrix)
    """
    labels_eb = build_clusters(deb_matrix, medoids)
    labels_bb = build_clusters(dbb_matrix, medoids)

    xbeb = xie_beni_eb(deb_matrix, medoids, labels_eb)
    xbbb = xie_beni_eb(dbb_matrix, medoids, labels_bb)

    return xbeb, xbbb


def evaluate_population_objectives(
    population: np.ndarray,
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
) -> np.ndarray:
    """
    Evalúa (XBEB, XBBB) para cada individuo de una población.

    Parámetros
    ----------
    population : np.ndarray — población de medoides, shape (pop_size, k).
    deb_matrix  : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    dbb_matrix  : np.ndarray — matriz (n×n) de distancia biológica (DBB).

    Retorna
    -------
    np.ndarray de shape (pop_size, 2), columna 0 = XBEB, columna 1 = XBBB.

    Ejemplo
    -------
    >>> objectives = evaluate_population_objectives(population, deb, dbb)
    >>> print(objectives.shape)  # (pop_size, 2)
    """
    pop_size = population.shape[0]
    objectives = np.empty((pop_size, 2), dtype=np.float64)

    for i, medoids in enumerate(population):
        objectives[i] = evaluate_objectives(medoids, deb_matrix, dbb_matrix)

    return objectives


# ══════════════════════════════════════════════════════════════════════════════
# 2. NON-DOMINATED SORTING
# ══════════════════════════════════════════════════════════════════════════════

def _dominates(obj_a, obj_b) -> bool:
    """
    Determina si la solución A domina a la solución B (minimización),
    según la definición (2) del paper:

        A ≺ B  ⟺  ∀t: P_t(A) ≤ P_t(B)  ∧  ∃t: P_t(A) < P_t(B)

    Parámetros
    ----------
    obj_a, obj_b : secuencias de floats (ej. (XBEB, XBBB)).

    Retorna
    -------
    bool — True si A domina a B.
    """
    not_worse = all(a <= b for a, b in zip(obj_a, obj_b))
    strictly_better = any(a < b for a, b in zip(obj_a, obj_b))
    return not_worse and strictly_better


def non_dominated_sort(objectives) -> list[list[int]]:
    """
    Ordena una población en frentes de Pareto (F1, F2, ...) según dominancia,
    replicando el procedimiento de NSGA-II.

    Parámetros
    ----------
    objectives : np.ndarray de shape (pop_size, 2), o lista de tuplas
                 (XBEB, XBBB) — un par de objetivos por individuo.

    Retorna
    -------
    list[list[int]] — lista de frentes; cada frente es una lista de índices
    (posiciones en `objectives`). F1 = fronts[0] son las soluciones no
    dominadas, F2 = fronts[1] las siguientes, etc.

    Ejemplo
    -------
    >>> fronts = non_dominated_sort(objectives)
    >>> f1 = fronts[0]  # individuos no dominados
    """
    n = len(objectives)
    domination_count = [0] * n
    dominated_by = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # descartar el último frente vacío


# ══════════════════════════════════════════════════════════════════════════════
# 3. CROWDING DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def crowding_distance(objectives, front: list[int]) -> dict[int, float]:
    """
    Calcula la crowding distance de las soluciones de un frente de Pareto
    para 2 objetivos.

    Las soluciones en los extremos de cada objetivo reciben distancia
    infinita (se priorizan para mantener diversidad). Si el frente tiene
    1 o 2 elementos, todas reciben distancia infinita.

    Parámetros
    ----------
    objectives : np.ndarray de shape (pop_size, 2), o lista de tuplas
                 (XBEB, XBBB) — todos los objetivos de la población.
    front      : list[int] — índices del frente a evaluar.

    Retorna
    -------
    dict[int, float] — mapeo índice → crowding distance.

    Ejemplo
    -------
    >>> fronts = non_dominated_sort(objectives)
    >>> dist = crowding_distance(objectives, fronts[0])
    """
    distances = {idx: 0.0 for idx in front}
    size = len(front)

    if size <= 2:
        return {idx: np.inf for idx in front}

    for obj_idx in range(2):  # XBEB, XBBB
        sorted_front = sorted(front, key=lambda i: objectives[i][obj_idx])

        f_min = objectives[sorted_front[0]][obj_idx]
        f_max = objectives[sorted_front[-1]][obj_idx]

        distances[sorted_front[0]] = np.inf
        distances[sorted_front[-1]] = np.inf

        if f_max == f_min:
            continue

        denom = f_max - f_min
        for i in range(1, size - 1):
            idx = sorted_front[i]
            if distances[idx] == np.inf:
                continue
            prev_val = objectives[sorted_front[i - 1]][obj_idx]
            next_val = objectives[sorted_front[i + 1]][obj_idx]
            contribution = (next_val - prev_val) / denom
            if not np.isfinite(contribution):
                contribution = 0.0
            distances[idx] += contribution

    return distances


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE: NON-DOMINATEDSORTING-CROWDINGDISTANCE (Algoritmo 1, líneas 9/12/14)
# ══════════════════════════════════════════════════════════════════════════════

def non_dominated_sorting_crowding_distance(objectives) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Aplica el procedimiento NON-DOMINATEDSORTING-CROWDINGDISTANCE de
    NSGA-II (Algoritmo 1, líneas 9, 12 y 14 del paper) sobre una población
    completa.

    Asigna a cada individuo:
        - rank: número de frente al que pertenece (1 = F1 = no dominados).
        - crowding distance: dispersión dentro de su frente.

    Este resultado es el insumo estándar para:
        - Selección por torneo binario (ranking + crowding distance).
        - Truncamiento de Rv a tamaño N (se descartan los peores ranks,
          y dentro del último rank incluido, los de menor crowding distance).

    Parámetros
    ----------
    objectives : np.ndarray de shape (pop_size, 2), o lista de tuplas
                 (XBEB, XBBB).

    Retorna
    -------
    ranks    : np.ndarray de shape (pop_size,), dtype int — rank de cada
               individuo (1 = F1, 2 = F2, ...).
    crowding : np.ndarray de shape (pop_size,), dtype float — crowding
               distance de cada individuo dentro de su frente.
    fronts   : list[list[int]] — frentes de Pareto (igual que
               non_dominated_sort()).

    Ejemplo
    -------
    >>> ranks, crowding, fronts = non_dominated_sorting_crowding_distance(objectives)
    >>> print(f"Individuos en F1: {fronts[0]}")
    >>> print(f"Rank del individuo 3: {ranks[3]}, crowding: {crowding[3]:.4f}")
    """
    n = len(objectives)
    fronts = non_dominated_sort(objectives)

    ranks = np.empty(n, dtype=np.int32)
    crowding = np.empty(n, dtype=np.float64)

    for rank, front in enumerate(fronts, start=1):
        cd = crowding_distance(objectives, front)
        for idx in front:
            ranks[idx] = rank
            crowding[idx] = cd[idx]

    return ranks, crowding, fronts


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIÓN: FRENTES DE PARETO EN UN PLANO 2D
# ══════════════════════════════════════════════════════════════════════════════

def plot_pareto_fronts(
    objectives,
    fronts: list[list[int]] | None = None,
    output_dir: str | None = None,
    filename: str = "pareto_fronts.png",
    xlabel: str = "XBEB (expresión)",
    ylabel: str = "XBBB (biológica)",
    title: str = "Frentes de Pareto",
    max_fronts_highlighted: int = 5,
    show: bool = False,
) -> str | None:
    """
    Grafica los frentes de Pareto de una población en un plano bidimensional
    (XBEB vs XBBB), coloreando cada frente con un color distinto.

    El frente F1 (no dominado) se conecta además con una línea punteada
    para visualizar la forma del frente.

    Soluciones con valores no finitos (XBEB o XBBB = inf, correspondientes
    a soluciones degeneradas) se excluyen del gráfico.

    Parámetros
    ----------
    objectives             : np.ndarray de shape (pop_size, 2), o lista de
                              tuplas (XBEB, XBBB).
    fronts                 : list[list[int]] — frentes de Pareto, salida de
                              non_dominated_sort(). Si es None, se calculan
                              internamente.
    output_dir             : str — carpeta donde guardar el gráfico (se crea
                              si no existe). Si es None, no se guarda archivo.
    filename               : str — nombre del archivo de imagen.
    xlabel, ylabel, title  : str — etiquetas del gráfico.
    max_fronts_highlighted : int — número máximo de frentes que aparecen en
                              la leyenda con etiqueta propia (F1, F2, ...);
                              el resto se grafican sin entrada en la leyenda.
    show                   : bool — si True, muestra el gráfico interactivamente
                              (plt.show()). Por defecto False (entornos sin GUI).

    Retorna
    -------
    str | None — ruta del archivo guardado, o None si output_dir es None.

    Ejemplo
    -------
    >>> ranks, crowding, fronts = non_dominated_sorting_crowding_distance(objectives)
    >>> plot_pareto_fronts(objectives, fronts=fronts, output_dir="resultados")
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objectives = np.asarray(objectives, dtype=np.float64)

    if fronts is None:
        fronts = non_dominated_sort(objectives)

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap("viridis")
    n_fronts = len(fronts)

    for rank, front in enumerate(fronts):
        pts = objectives[front]
        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts_finite = pts[finite_mask]

        if len(pts_finite) == 0:
            continue

        color = cmap(rank / max(n_fronts - 1, 1))
        label = f"F{rank + 1}" if rank < max_fronts_highlighted else None

        ax.scatter(
            pts_finite[:, 0], pts_finite[:, 1],
            color=color, label=label,
            s=45, edgecolors="black", linewidths=0.5,
            zorder=n_fronts - rank,
        )

        # Conectar el frente F1 con una línea punteada para visualizar su forma.
        if rank == 0 and len(pts_finite) > 1:
            order = np.argsort(pts_finite[:, 0])
            ax.plot(
                pts_finite[order, 0], pts_finite[order, 1],
                linestyle="--", color=color, alpha=0.6, zorder=0,
            )

    n_excluded = int(np.sum(~np.all(np.isfinite(objectives), axis=1)))
    if n_excluded > 0:
        title = f"{title}\n({n_excluded} soluciones degeneradas excluidas, XB=inf)"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small", title="Frentes")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    saved_path = None
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_path = out_path / filename
        fig.savefig(saved_path, dpi=150)
        print(f"[save] Gráfico de frentes de Pareto guardado en: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(saved_path) if saved_path is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE COMPLETO: EVALUAR + ORDENAR + GUARDAR + GRAFICAR
# ══════════════════════════════════════════════════════════════════════════════

def run_population_sorting(
    population: np.ndarray,
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    output_dir: str,
    gene_names: list[str] | None = None,
    prefix: str = "population",
    plot: bool = True,
) -> dict:
    """
    Pipeline completo: evalúa (XBEB, XBBB) para una población, aplica
    non-dominated sorting + crowding distance, guarda una tabla resumen
    y opcionalmente grafica los frentes de Pareto.

    Parámetros
    ----------
    population : np.ndarray — población de medoides, shape (pop_size, k).
    deb_matrix  : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    dbb_matrix  : np.ndarray — matriz (n×n) de distancia biológica (DBB).
    output_dir  : str        — carpeta de salida.
    gene_names  : list[str]  — nombres de genes (opcional). Si se provee,
                  las columnas de medoides muestran nombres de genes en
                  vez de índices.
    prefix      : str        — prefijo para los archivos de salida.
    plot        : bool       — si True, genera y guarda el gráfico de
                  frentes de Pareto.

    Retorna
    -------
    dict con claves:
        "dataframe" : pd.DataFrame — tabla con medoides, XBEB, XBBB, rank
                       y crowding distance por individuo.
        "objectives": np.ndarray — shape (pop_size, 2).
        "ranks"     : np.ndarray — shape (pop_size,).
        "crowding"  : np.ndarray — shape (pop_size,).
        "fronts"    : list[list[int]].
        "csv_path"  : str — ruta del CSV guardado.
        "plot_path" : str | None — ruta del gráfico guardado.

    Ejemplo
    -------
    >>> result = run_population_sorting(
    ...     population=population,
    ...     deb_matrix=deb, dbb_matrix=dbb,
    ...     output_dir="resultados", gene_names=gene_names,
    ... )
    >>> print(result["dataframe"].head())
    """
    pop_size, k = population.shape

    print(f"[sorting] Evaluando (XBEB, XBBB) para {pop_size} individuos...")
    objectives = evaluate_population_objectives(population, deb_matrix, dbb_matrix)

    print("[sorting] Aplicando non-dominated sorting + crowding distance...")
    ranks, crowding, fronts = non_dominated_sorting_crowding_distance(objectives)

    print(f"[sorting] Número de frentes: {len(fronts)}")
    for i, front in enumerate(fronts, start=1):
        print(f"  F{i}: {len(front)} individuos")

    # ── Tabla resumen ───────────────────────────────────────────────────────
    rows = []
    for i in range(pop_size):
        row = {"individuo": f"solucion{i + 1}"}
        for m in range(k):
            val = population[i, m]
            row[f"medoid_{m + 1}"] = gene_names[val] if gene_names is not None else int(val)
        row["XBEB"] = objectives[i, 0]
        row["XBBB"] = objectives[i, 1]
        row["rank"] = int(ranks[i])
        row["crowding_distance"] = crowding[i]
        rows.append(row)

    df = pd.DataFrame(rows)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{prefix}_sorting.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] Tabla de ordenamiento guardada en: {csv_path}")

    # ── Gráfico ─────────────────────────────────────────────────────────────
    plot_path = None
    if plot:
        plot_path = plot_pareto_fronts(
            objectives, fronts=fronts,
            output_dir=output_dir, filename=f"{prefix}_pareto_fronts.png",
        )

    return {
        "dataframe": df,
        "objectives": objectives,
        "ranks": ranks,
        "crowding": crowding,
        "fronts": fronts,
        "csv_path": str(csv_path),
        "plot_path": plot_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent))
    from solution_definition import load_distance_matrix, random_medoids_pop

    deb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBE.csv"
    dbb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBI.csv"
    output_dir = r"C:\Users\benja\Desktop\workspace\Thesis\Results"
    k = 4
    pop_size = 50
    seed = 42

    deb, gene_names = load_distance_matrix(deb_path)
    dbb, _ = load_distance_matrix(dbb_path)
    n = len(gene_names)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)

    result = run_population_sorting(
        population=population,
        deb_matrix=deb, dbb_matrix=dbb,
        output_dir=output_dir,
        gene_names=gene_names,
        prefix="initial_population",
    )

    print("\n── Primeras filas de la tabla ──")
    print(result["dataframe"].head(10).to_string(index=False))