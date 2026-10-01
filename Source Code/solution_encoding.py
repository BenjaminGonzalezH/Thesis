"""
Codificación de soluciones (arreglo de medoides), carga de matrices de
distancia y función objetivo (Xie-Beni) para el NSGA-II de clustering
de genes.
"""

############################
# Importaciones
############################
import numpy as np              # Operaciones matemáticas.

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
    D   = xp.asarray(distance_matrix)
    pop = xp.asarray(population)

    # Submatriz de distancia que representa la distancia de cada gen
    # de la población a los respectivos medoides de cada uno de los
    # arreglos de medoides para la asignación.
    # Coordenadas: [gen, solución, medoide].
    D_pop = D[:, pop]
    
    # Cluster más cercano para cada elemento, por individuo (0-based).
    # Coordenadas: [gen, indice de medoide].
    labels_xp = xp.argmin(D_pop, axis=-1)

    # Reordenar a (pop_size, n) y pasar a 1-based.
    # Coordenadas: [solución, asignación de cluster]
    # Se cambia la asignación a <indice de cluster> + 1.
    labels_xp = labels_xp.T                 # (pop_size, n)

    if _GPU:
        return cp.asnumpy(labels_xp).astype(np.int32) + 1
    return labels_xp.astype(np.int32) + 1


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
    tuple[float, float] — (XBEB, XBBB). Valores menores indican mejor solución.
    Cada componente retorna np.inf si dos medoides son idénticos en esa matriz.
    """
    xp = cp if _GPU else np

    # Declaración de variable como versión global.
    global OBJ_FUNCTION_CALLS

    # Transformación de elementos a interfaz GPU.
    D_stack = xp.stack([xp.asarray(ge_matrix), xp.asarray(bi_matrix)])      # Fusión de ambas matrices de distancia (Matriz, n, n).
    med     = xp.asarray(medoids)                                           # Población de medoides asociada (sol, medoides).
    n       = D_stack.shape[1]                                              # Cantidad de genes.
    k       = med.shape[0]                                                  # Cantidad de medoides por solución.

    # Elementos asociado a los clusters.
    cluster_idx     = labels - 1                                            # (n,)   Solución de clustering - 0-based
    medoid_per_elem = med[cluster_idx]                                      # (n,)   Asociación gen-medoide basado en el índice del gen.
    idx_mat         = xp.arange(2)[:, None]                                 # (2, 1) Indices de las dos matrices (0: expr ; 1: info).
    idx_elem        = xp.arange(n)[None, :]                                 # (1, n) Indices asociados a los elementos (genes).

    # Cálculo de distancia entre el medoide asignado según cada elemento
    # de los clusters (ambas matrices) y definición del numerador.
    dist_to_medoid  = D_stack[idx_mat, idx_elem, xp.broadcast_to(medoid_per_elem, (2, n))]
    numerator       = xp.sum(dist_to_medoid ** 2, axis=1)   # (2,)

    # Calculo del denominador (menor distancia entre clusters)
    D_med     = D_stack[:, med, :][:, :, med]                               # Sub matriz considerando los medoides.
    diag_mask = xp.eye(k, dtype=bool)                                       # Marca de diagonal (ignorar al ser 0).
    D_med_off = xp.where(diag_mask, xp.inf, D_med)                          # Aplicar infinito a las diagonales.
    min_inter = xp.min(D_med_off ** 2, axis=(1, 2))                         # Par de medoides con menor distancia.

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

    # Transformart elementos a interfaz GPU.
    D_stack = xp.stack([xp.asarray(ge_matrix), xp.asarray(bi_matrix)])      # Fusión de matrices de distancia (Matriz, n, n).
    pop = xp.asarray(population)                                            # Población de soluciones medoids array (pop_size, k)
    lbl = xp.asarray(labels_pop)                                            # Población de soluciones labels array (pop_size, n)
    pop_size, k = pop.shape                                                 # Número de soluciones, grupos y genes.
    n = D_stack.shape[1]

    # Elementos asociados a los clusters.
    cluster_idx = lbl - 1                                                   # Soluciones de clutering en formato 0-based.
    medoid_per_elem = xp.take_along_axis(pop, cluster_idx, axis=1)          # Aociación gen-medoide (versión globalizada en la población).
    idx_mat  = xp.arange(2)[:, None, None]                                  # Indices de las dos matrices (0: expr ; 1: info).
    idx_elem = xp.arange(n)[None, None, :]                                  # Indices asociados a los elementos (genes).

    # Cálculo de distancia entre el medoide asignado según cada elemento
    # de los clusters (ambas matrices) y definición del numerador.
    dist_to_medoid = D_stack[idx_mat, idx_elem, medoid_per_elem[None, :, :]]   # (2, pop_size, n)
    numerator = xp.sum(dist_to_medoid ** 2, axis=2)                            # (2, pop_size)

    # Calculo de denominador: D_med por individuo: se necesita un índice 
    # de "individuo" explícito para no cruzar medoides entre individuos 
    # distintos.
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