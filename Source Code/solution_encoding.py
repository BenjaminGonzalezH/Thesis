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
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
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

    D_stack = xp.stack([xp.asarray(deb_matrix), xp.asarray(dbb_matrix)])
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

    xb_deb, xb_dbb = numerator / (n * min_inter)
    xb_deb = np.inf if min_inter[0] == 0.0 else float(xb_deb)
    xb_dbb = np.inf if min_inter[1] == 0.0 else float(xb_dbb)

    OBJ_FUNCTION_CALLS += 1

    return xb_deb, xb_dbb