"""
Codificación de soluciones (arreglo de medoides), carga de matrices de
distancia y función objetivo (Xie-Beni) para el NSGA-II de clustering
de genes.
"""

############################
# Importaciones
############################
import numpy as np              # Operaciones matemáticas.
from collections import OrderedDict

############################
# Variables globales
############################
OBJ_FUNCTION_CALLS = 0          # Administra las llamadas a función objetivo (criterio de paro).

##################################
# Configuraciones
##################################
# Se da opción de ocupar GPU para administrar operaciones vectorizadas
# y operadas por ese hardware en caso de ser necesario.
try:
    import cupy as cp
    _GPU = True
    print("[backend] CuPy detectado → operaciones vectorizadas en GPU.")
except ImportError:
    cp = np
    _GPU = False
    print("[backend] CuPy no encontrado → usando NumPy (CPU).")


##################################
# Caché de matrices en el backend activo
##################################
# Las matrices de distancia (n×n) son constantes durante una ejecución, pero las
# funciones objetivo se invocan miles de veces (con lotes pequeños en PR/PLS).
# Convertirlas en cada llamada (transferencia host→device en GPU, o copia
# 2·n² por apilado en CPU) dominaba el costo. Se conservan en un caché LRU
# pequeño indexado por identidad del arreglo (se guarda la referencia al
# original, así el id no puede reutilizarse mientras la entrada exista).
# Supuesto: las matrices NO se mutan in-place durante la ejecución; si se
# hiciera, llamar a clear_device_cache().
_DEVICE_CACHE = OrderedDict()
_DEVICE_CACHE_SIZE = 8

# Tope (bytes) del tensor temporal (n, bloque, k) de build_clusters_population.
_CLUSTER_CHUNK_BYTES = 128 * 1024 * 1024

def _as_backend(matrix):
    """Devuelve `matrix` en el backend activo, reutilizando el caché."""
    if not _GPU:
        return np.asarray(matrix)            # CPU: sin copia.
    key = id(matrix)
    hit = _DEVICE_CACHE.get(key)
    if hit is not None and hit[0] is matrix:
        _DEVICE_CACHE.move_to_end(key)
        return hit[1]
    dev = cp.asarray(matrix)
    _DEVICE_CACHE[key] = (matrix, dev)
    if len(_DEVICE_CACHE) > _DEVICE_CACHE_SIZE:
        _DEVICE_CACHE.popitem(last=False)
    return dev

def clear_device_cache():
    """Vacía el caché (usar si una matriz se modifica in-place)."""
    _DEVICE_CACHE.clear()


##################################
# Conjunto de funciones.
##################################

def load_distance_matrix(filepath):
    """
    Carga de matrices de distancias en versión comprimioda.
    """
    # Carga de los datos dentro del archivo.
    data = np.load(filepath, allow_pickle=True)
    genes = data["genes"].tolist()

    n = len(genes)                              # Orden de la matriz.
    iu = np.triu_indices(n, k=1)                # índices de la triangular superior.
    M = np.zeros((n, n), dtype=np.float32)      # Matriz nxn.

    # Carga de datos y completar valores de triangular inferior (como diagonal).
    M[iu] = data["upper"]
    M += M.T
    np.fill_diagonal(M, data["diag"])

    # Retroalimentación y sálida.
    print(f"[load] '{filepath.name}' → matriz {n}×{n}. Primer gen: '{genes[0]}', último: '{genes[-1]}'.")
    return 1 - M, genes

def random_medoids_pop(n: int, k: int, pop_size: int = 1, seed: int | None = None) -> np.ndarray:
    """
    Construcción de la población inicial a través del uso de de funciones psudoaleatorias
    de construcción de arreglos.
    """
    # Se inicializa la interfaz de seleccuión de número psudoaleatorios.
    rng = np.random.default_rng(seed)

    # Generación de 'n' arreglos de tamaño 'k' mediante la selección del
    # índice asociado al gen. 
    population = np.stack(
        [rng.choice(n, size=k, replace=False) for _ in range(pop_size)]
    ).astype(np.int32)

    return population

def build_clusters_population(
    distance_matrix: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """
    Construcción de matrices del formato de medoides al formato de labels,
    para una población completa de individuos en una sola operación vectorizada.
    """
    xp  = cp if _GPU else np          # backend activo

    # Transformación de elementos numpy a matrices manejables por GPU, siendo
    # su equivalente, es decir, D es una matriz cuadrada de orden 'g' genes y 
    # population es un arreglo de 'n' soluciones de 'k' medoides cada una.
    D   = _as_backend(distance_matrix)
    pop = xp.asarray(population)

    # Submatriz de distancia gen→medoides, coordenadas [gen, solución, medoide],
    # de tamaño n·m·k. Para acotar el pico de memoria se procesa la población
    # por bloques de individuos (ver _CLUSTER_CHUNK_BYTES); el resultado es
    # idéntico al cálculo en un solo paso (el argmin es independiente por individuo).
    n_genes = D.shape[0]
    m, k = pop.shape
    chunk = max(1, _CLUSTER_CHUNK_BYTES // (n_genes * k * D.dtype.itemsize))

    # Cluster más cercano para cada elemento, por individuo (0-based).
    # Coordenadas: [solución, gen]. Se usa int32 desde el origen (argmin da int64).
    labels_xp = xp.empty((m, n_genes), dtype=xp.int32)
    for i in range(0, m, chunk):
        D_pop = D[:, pop[i:i + chunk]]                      # (n, c, k)
        labels_xp[i:i + chunk] = xp.argmin(D_pop, axis=-1).T
        del D_pop

    # Se cambia la asignación a <indice de cluster> + 1 (1-based).
    if _GPU:
        return cp.asnumpy(labels_xp) + 1
    return labels_xp + 1


def xie_beni(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    medoids: np.ndarray,
    labels:  np.ndarray,
) -> tuple[float, float]:
    """
    Calcula los índices Xie-Beni de expresión (XB_GE) y de información (XB_BI)
    para una solución individual, en una sola operación vectorizada.

    Considerar que la fórmula de Xie-beni (XB) definida en el trabajo de referencia es:
    XB(C) = (Sumatoria TODAS las distancias al cuadrado entre los elementos de un cluster a su respectivo medoide)
     / (numero total de elementos) por (Distancia minima entre un par de medoides).

    Parámetros
    ----------
    ge_matrix : np.ndarray — matriz (n×n) de distancias de expresión   (GE).
    bi_matrix : np.ndarray — matriz (n×n) de distancias de información (BI).
    medoids    : np.ndarray — arreglo 1-D de k índices de medoides (0-based).
    labels     : np.ndarray — labels (1-based) de build_clusters(dge_matrix, medoids).

    Retorna
    -------
    tuple[float, float] — (XB_GE, XB_BI). Valores menores indican mejor solución.
    Cada componente retorna np.inf si dos medoides son idénticos en esa matriz.
    """
    xp = cp if _GPU else np

    # Declaración de variable como versión global.
    global OBJ_FUNCTION_CALLS

    # Matrices en el backend activo (caché; sin apilar ni copiar).
    Ds      = (_as_backend(ge_matrix), _as_backend(bi_matrix))              # (GE, BI), cada una (n, n).
    med     = xp.asarray(medoids)                                           # Medoides de la solución (k,).
    n       = Ds[0].shape[0]                                                # Cantidad de genes.
    k       = med.shape[0]                                                  # Cantidad de medoides.

    # Elementos asociado a los clusters.
    cluster_idx     = xp.asarray(labels) - 1                                # (n,)   0-based
    medoid_per_elem = med[cluster_idx]                                      # (n,)   Medoide asignado a cada gen.
    idx_elem        = xp.arange(n)
    diag_mask       = xp.eye(k, dtype=bool)

    # Numerador (suma de distancias² al medoide) y denominador (menor
    # distancia² entre medoides), por cada matriz.
    numerator = xp.stack([xp.sum(D[idx_elem, medoid_per_elem] ** 2) for D in Ds])                  # (2,)
    min_inter = xp.stack([xp.min(xp.where(diag_mask, xp.inf, D[med][:, med]) ** 2) for D in Ds])   # (2,)

    # Calculo vectorizado por GPU.
    if _GPU:
        numerator = cp.asnumpy(numerator)
        min_inter = cp.asnumpy(min_inter)

    # Cálculo del índice de Xie-beni de la solución considerando ambas matrices.
    xb_ge, xb_bi = numerator / (n * min_inter)
    xb_ge = np.inf if min_inter[0] == 0.0 else float(xb_ge)
    xb_bi = np.inf if min_inter[1] == 0.0 else float(xb_bi)

    # Conteo de llamas a función objetivo.
    OBJ_FUNCTION_CALLS += 1

    return xb_ge, xb_bi

def xie_beni_population(ge_matrix, bi_matrix, population, labels_pop):
    """
    Versión vectorizada de la función anterior para calcular xie-beni de la población.
    """
    # Backend GPU activo y seteo global de variable de control de llamadas a función objetivo.
    xp = cp if _GPU else np
    global OBJ_FUNCTION_CALLS

    # Matrices en el backend activo (caché; sin apilar ni copiar).
    Ds  = (_as_backend(ge_matrix), _as_backend(bi_matrix))                  # (GE, BI), cada una (n, n).
    pop = xp.asarray(population)                                            # Población de medoides (pop_size, k).
    lbl = xp.asarray(labels_pop)                                            # Labels (pop_size, n), 1-based.
    pop_size, k = pop.shape
    n = Ds[0].shape[0]

    # Asociación gen-medoide por individuo.
    medoid_per_elem = xp.take_along_axis(pop, lbl - 1, axis=1)              # (pop_size, n)
    idx_elem = xp.arange(n)[None, :]                                        # (1, n)
    diag_mask = xp.eye(k, dtype=bool)[None, :, :]
    idx_row = pop[:, :, None]                                               # (pop_size, k, 1)
    idx_col = pop[:, None, :]                                               # (pop_size, 1, k)

    # Numerador: suma de distancias² gen→medoide. Denominador: menor
    # distancia² entre medoides DEL MISMO individuo (índice explícito para
    # no cruzar medoides entre individuos).
    numerator = xp.stack([xp.sum(D[idx_elem, medoid_per_elem] ** 2, axis=1) for D in Ds])          # (2, pop_size)
    min_inter = xp.stack([xp.min(xp.where(diag_mask, xp.inf, D[idx_row, idx_col]) ** 2, axis=(1, 2))
                          for D in Ds])                                                            # (2, pop_size)

    if _GPU:
        numerator, min_inter = cp.asnumpy(numerator), cp.asnumpy(min_inter)

    degenerate = min_inter == 0.0
    xb = np.where(degenerate, np.inf, numerator / (n * np.where(degenerate, 1.0, min_inter)))

    OBJ_FUNCTION_CALLS += pop_size
    return xb.T   # (pop_size, 2)