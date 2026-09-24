"""
Codificación de soluciones (arreglo de medoides), carga de matrices de
distancia y función objetivo (Xie-Beni) para el NSGA-II de clustering
de genes.
"""

############################
# Importaciones
############################
import numpy as np
import pandas as pd
from pathlib import Path

############################
# Variables globales
############################
OBJ_FUNCTION_CALLS = 0   # Administra las llamadas a función objetivo (criterio de paro).

##################################
# Configuraciones
##################################
try:
    import cupy as cp
    _GPU = True
    print("[backend] CuPy detectado → operaciones vectorizadas en GPU.")
except ImportError:
    cp = np
    _GPU = False
    print("[backend] CuPy no encontrado → usando NumPy (CPU).")


def load_distance_matrix(filepath: Path) -> tuple[np.ndarray, list[str]]:
    """
    Lee los archivos csv generados en la etapa de pre-procesamiento de los datasets,
    cambiando su formato de similitud a matriz de distancia.
    """
    df = pd.read_csv(filepath, index_col=0)
    gene_names = df.index.astype(str).tolist()
    matrix = 1 - df.values.astype(np.float64)

    print(
        f"[load] '{filepath.name}' → matriz {matrix.shape[0]}×{matrix.shape[1]} cargada. "
        f"Primer gen: '{gene_names[0]}', último: '{gene_names[-1]}'."
    )
    return matrix, gene_names


def random_medoids_pop(n: int, k: int, pop_size: int = 1, seed: int | None = None) -> np.ndarray:
    """Construcción de la población inicial mediante selección pseudoaleatoria de índices."""
    rng = np.random.default_rng(seed)
    population = np.stack(
        [rng.choice(n, size=k, replace=False) for _ in range(pop_size)]
    ).astype(np.int32)
    return population


def build_clusters_population(
    distance_matrix: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """Construcción vectorizada de labels (clusters) para toda una población de medoides."""
    xp = cp if _GPU else np

    D = xp.asarray(distance_matrix)
    pop = xp.asarray(population)

    D_pop = D[:, pop]                      # (n, pop_size, k)
    labels_xp = xp.argmin(D_pop, axis=-1)  # (n, pop_size)
    labels_xp = labels_xp.T                # (pop_size, n)

    if _GPU:
        return cp.asnumpy(labels_xp).astype(np.int32) + 1
    return labels_xp.astype(np.int32) + 1


def xie_beni(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    medoids: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Calcula los índices Xie-Beni de expresión (XB_GE) y de información (XB_BI)
    para una solución individual, en una sola operación vectorizada.

    XB(C) = (Σ distancias² de cada elemento a su medoide) / (n · distancia² mínima entre medoides).
    """
    xp = cp if _GPU else np

    global OBJ_FUNCTION_CALLS

    D_stack = xp.stack([xp.asarray(ge_matrix), xp.asarray(bi_matrix)])
    med = xp.asarray(medoids)
    n = D_stack.shape[1]
    k = med.shape[0]

    cluster_idx = labels - 1
    medoid_per_elem = med[cluster_idx]
    idx_mat = xp.arange(2)[:, None]
    idx_elem = xp.arange(n)[None, :]

    dist_to_medoid = D_stack[idx_mat, idx_elem, xp.broadcast_to(medoid_per_elem, (2, n))]
    numerator = xp.sum(dist_to_medoid ** 2, axis=1)

    D_med = D_stack[:, med, :][:, :, med]
    diag_mask = xp.eye(k, dtype=bool)
    D_med_off = xp.where(diag_mask, xp.inf, D_med)
    min_inter = xp.min(D_med_off ** 2, axis=(1, 2))

    if _GPU:
        numerator = cp.asnumpy(numerator)
        min_inter = cp.asnumpy(min_inter)

    xb_ge, xb_bi = numerator / (n * min_inter)
    xb_ge = np.inf if min_inter[0] == 0.0 else float(xb_ge)
    xb_bi = np.inf if min_inter[1] == 0.0 else float(xb_bi)

    OBJ_FUNCTION_CALLS += 1

    return xb_ge, xb_bi

def xie_beni_population(ge_matrix, bi_matrix, population, labels_pop):
    xp = cp if _GPU else np
    global OBJ_FUNCTION_CALLS

    D_stack = xp.stack([xp.asarray(ge_matrix), xp.asarray(bi_matrix)])  # 1 vez, no por individuo
    pop = xp.asarray(population)          # (pop_size, k)
    lbl = xp.asarray(labels_pop)          # (pop_size, n)
    pop_size, k = pop.shape
    n = D_stack.shape[1]

    cluster_idx = lbl - 1
    medoid_per_elem = xp.take_along_axis(pop, cluster_idx, axis=1)   # (pop_size, n)

    idx_mat  = xp.arange(2)[:, None, None]
    idx_elem = xp.arange(n)[None, None, :]
    dist_to_medoid = D_stack[idx_mat, idx_elem, medoid_per_elem[None, :, :]]   # (2, pop_size, n)
    numerator = xp.sum(dist_to_medoid ** 2, axis=2)                            # (2, pop_size)

    # D_med por individuo: se necesita un índice de "individuo" explícito para no
    # cruzar medoides entre individuos distintos.
    idx_row = pop[None, :, :, None]     # (1, pop_size, k, 1)
    idx_col = pop[None, :, None, :]     # (1, pop_size, 1, k)
    D_med = D_stack[xp.arange(2)[:, None, None, None], idx_row, idx_col]       # (2, pop_size, k, k)

    diag_mask = xp.eye(k, dtype=bool)[None, None, :, :]
    D_med_off = xp.where(diag_mask, xp.inf, D_med)
    min_inter = xp.min(D_med_off ** 2, axis=(2, 3))                            # (2, pop_size)

    if _GPU:
        numerator, min_inter = cp.asnumpy(numerator), cp.asnumpy(min_inter)

    degenerate = min_inter == 0.0
    xb = np.where(degenerate, np.inf, numerator / (n * np.where(degenerate, 1.0, min_inter)))

    OBJ_FUNCTION_CALLS += pop_size
    return xb.T   # (pop_size, 2)