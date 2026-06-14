"""
Código fuente asociado a la construcción de representación de la solución, considerando los
siguientes componentes:
    1. Construcción de arreglo de medoides utilizando elementos reales del dataset como elementos
    de referencia.
    2. Lectura de las matrices DEB y DBB al estar pre-construidas, planificando que estas sean
    replicables en el contexto de la investigación como reutilizables.
    3. Construcción de grupos/clusters a partir de las matrices de distancia, dado a la compatibilidad
    con gclusters_analyzer se puede aplicar un dataframe de labels.
    4. Guardado de grupos resultantes.

    ¿Por qué se opta por GPU? Al presentarse todo como cálculo de elementos matemáticos, estos puede
    vectorizarse con el fin de aplicar paralelismo de forma nativa sin sobrecargar el código.
"""

##################################
# Imports
##################################
import numpy as np                                          # Aplicación de operaciones matematicas eficientes.
import pandas as pd                                         # Lectura de conjuntos de datos en csv.
from pathlib import Path                                    # Aplicar un formato generalizado de direcciones en el script.

##################################
# Configuraciones
##################################
# Se realiza la verificación de que sea posible el uso de CuPy
# con el fin de priorizar cálculo paralelo.
try:
    import cupy as cp
    _GPU = True
    print("[backend] CuPy detectado → operaciones vectorizadas en GPU.")
except ImportError:
    cp = np          # alias: mismo API que NumPy
    _GPU = False
    print("[backend] CuPy no encontrado → usando NumPy (CPU).")


##################################
# Funciones secundarias
##################################
def load_distance_matrix(filepath: str) -> tuple[np.ndarray, list[str]]:
    """
    Lee una matriz de distancia desde un archivo .csv.

    El formato esperado es:
        - Primera columna: nombres de genes (índice de filas).
        - Primera fila:    nombres de genes (encabezado de columnas).
        - Valores:         distancias float, diagonal = 0.

    Parámetros
    ----------
    filepath : str
        Ruta al archivo .csv.

    Retorna
    -------
    matrix : np.ndarray
        Matriz de distancia de forma (n, n) con dtype float64.
    gene_names : list[str]
        Nombres de los genes extraídos del índice de filas, en orden.

    Ejemplo
    -------
    >>> deb_matrix, gene_names = load_distance_matrix("matrices/DEB.csv")
    >>> dbb_matrix, _          = load_distance_matrix("matrices/DBB.csv")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Se esperaba un archivo .csv, se recibió: {path.suffix}")

    # La primera columna y fila siempre es el índice de genes, por ende, se puede
    # ignorar la primera columna y extraer el identificador de genes de la segunda.
    df = pd.read_csv(filepath, index_col=0)
    gene_names = df.index.astype(str).tolist()

    # Extracción de valores de las matrices.
    matrix = 1 - df.values.astype(np.float64)

    # Validaciones: Se espera una matriz cuadrada de distancia que provean un rango
    # de 1 (desiguales) hasta 0 (iguales).
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"La matriz leída no es cuadrada: shape={matrix.shape}. "
            "Verifica el formato del .csv."
        )
    if not np.allclose(np.diag(matrix), 0, atol=1e-6):
        raise ValueError(
            "La diagonal de la matriz no es ≈ 0. "
            "Verifica que la primera columna corresponda al índice de genes "
            "y no a valores de distancia."
        )

    print(
        f"[load] '{path.name}' → matriz {matrix.shape[0]}×{matrix.shape[1]} cargada. "
        f"Primer gen: '{gene_names[0]}', último: '{gene_names[-1]}'."
    )
    return matrix, gene_names


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINICIÓN DE ARREGLOS DE MEDOIDES (CROMOSOMA)
# ══════════════════════════════════════════════════════════════════════════════

def random_medoids_pop(n: int, k: int, pop_size: int = 1, seed: int | None = None) -> np.ndarray:
    """
    Genera una población aleatoria de arreglos de medoides.

    Parámetros
    ----------
    n        : int  — número de elementos en el dataset (genes).
    k        : int  — número de clusters deseado (medoides por individuo).
    pop_size : int  — tamaño de la población (número de individuos).
    seed     : int  — semilla para reproducibilidad (opcional).

    Retorna
    -------
    np.ndarray de shape (pop_size, k) donde cada fila contiene k índices
    únicos en [0, n-1] que representan los medoides de un individuo.

    Ejemplo
    -------
    >>> population = random_medoids_pop(n=384, k=4, pop_size=10, seed=42)
    >>> print(population.shape)  # (10, 4)
    >>> print(population[0])     # ej. [201  17 352  89]
    """
    if k > n:
        raise ValueError(f"k={k} no puede ser mayor que n={n}.")
    if pop_size < 1:
        raise ValueError(f"pop_size={pop_size} debe ser al menos 1.")
    rng = np.random.default_rng(seed)
    population = np.stack(
        [rng.choice(n, size=k, replace=False) for _ in range(pop_size)]
    ).astype(np.int32)
    return population


def medoids_from_list(indices: list[int], n: int) -> np.ndarray:
    """
    Crea un arreglo de medoides a partir de una lista de índices proporcionada
    explícitamente por el usuario o por el algoritmo.

    Parámetros
    ----------
    indices : list[int] — índices de los medoides (base 0).
    n       : int       — número total de elementos; sirve para validación.

    Retorna
    -------
    np.ndarray de shape (k,).

    Ejemplo
    -------
    >>> medoids = medoids_from_list([0, 5, 18, 82, 13, 2], n=100)
    """
    arr = np.array(indices, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("'indices' debe ser una lista 1-D.")
    if len(arr) != len(set(indices)):
        raise ValueError("Los índices de medoides deben ser únicos.")
    if arr.min() < 0 or arr.max() >= n:
        raise ValueError(f"Todos los índices deben estar en [0, {n-1}].")
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRUCCIÓN DE GRUPOS (CLUSTERS)
# ══════════════════════════════════════════════════════════════════════════════

def build_clusters(
    distance_matrix: np.ndarray,
    medoids: np.ndarray,
) -> np.ndarray:
    """
    Asigna cada elemento al cluster cuyo medoide esté más cercano,
    usando la matriz de distancia proporcionada.

    La operación central es vectorizable: se extrae la submatriz de
    distancias [n × k] y se aplica argmin sobre el eje de medoides.
    Si CuPy está disponible, el cómputo ocurre en GPU.

    Parámetros
    ----------
    distance_matrix : np.ndarray  — matriz (n×n) de distancias (DEB o DBB).
    medoids         : np.ndarray  — arreglo 1-D de k índices de medoides.

    Retorna
    -------
    np.ndarray de shape (n,) con el ID de cluster (1-based) de cada elemento.
    Se usa base 1 para coincidir con el formato de salida esperado.

    Ejemplo
    -------
    >>> labels = build_clusters(deb_matrix, medoids)
    """
    xp = cp if _GPU else np          # backend activo

    # Mover datos al backend
    D = xp.asarray(distance_matrix)  # (n, n)
    med = xp.asarray(medoids)        # (k,)

    # Submatriz de distancias a los medoides: shape (n, k)
    D_med = D[:, med]                # vectorizado, sin bucles Python

    # Cluster más cercano para cada elemento (0-based internamente)
    labels_xp = xp.argmin(D_med, axis=1)  # shape (n,)

    # Devolver siempre como NumPy, convirtiendo a 1-based
    if _GPU:
        return cp.asnumpy(labels_xp).astype(np.int32) + 1
    return labels_xp.astype(np.int32) + 1


def build_clusters_population(
    distance_matrix: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """
    Construye los clusters para toda una población de arreglos de medoides.

    Itera sobre cada individuo de la población y aplica build_clusters,
    produciendo una matriz de etiquetas donde cada fila corresponde a
    la asignación de clusters de un individuo.

    Parámetros
    ----------
    distance_matrix : np.ndarray — matriz (n×n) de distancias (DEB o DBB).
    population      : np.ndarray — matriz (pop_size×k) de medoides,
                                   salida de random_medoids_pop().

    Retorna
    -------
    np.ndarray de shape (pop_size, n) con IDs de cluster (1-based) por individuo.

    Ejemplo
    -------
    >>> all_labels = build_clusters_population(deb_matrix, population)
    >>> print(all_labels.shape)  # (pop_size, n)
    """
    pop_size = population.shape[0]
    n = distance_matrix.shape[0]
    all_labels = np.empty((pop_size, n), dtype=np.int32)

    for i, medoids in enumerate(population):
        all_labels[i] = build_clusters(distance_matrix, medoids)

    return all_labels


# ══════════════════════════════════════════════════════════════════════════════
# 4. GUARDAR GRUPOS EN DATAFRAME → .csv
# ══════════════════════════════════════════════════════════════════════════════

def save_clusters(
    all_labels: np.ndarray,
    gene_names: list[str],
    output_dir: str,
    filename: str = "clusters.csv",
) -> str:
    """
    Guarda la asignación de clusters de toda la población en un único .csv.

    Formato de salida
    -----------------
    Filas    : cada solución de la población (solucion1, solucion2, ...).
    Columnas : nombres de los genes.
    Valores  : ID de cluster asignado (1-based).

    Ejemplo de salida:
              gen1  gen2  gen3
    solucion1    2     1     3
    solucion2    1     3     2

    Parámetros
    ----------
    all_labels : np.ndarray — matriz (pop_size×n) de etiquetas, salida de
                              build_clusters_population().
    gene_names : list[str]  — nombres de genes extraídos de load_distance_matrix().
    output_dir : str        — carpeta destino (se crea si no existe).
    filename   : str        — nombre del archivo de salida.

    Retorna
    -------
    str — ruta completa del archivo guardado.

    Ejemplo
    -------
    >>> path = save_clusters(all_labels, gene_names, output_dir="resultados", filename="pop.csv")
    """
    pop_size, n = all_labels.shape

    if len(gene_names) != n:
        raise ValueError(
            f"gene_names tiene {len(gene_names)} entradas pero all_labels tiene {n} columnas."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Índice de filas: solucion1, solucion2, ...
    row_index = [f"solucion{i + 1}" for i in range(pop_size)]

    df = pd.DataFrame(all_labels, index=row_index, columns=gene_names)

    full_path = out_path / filename
    df.to_csv(full_path, index=True)
    print(f"[save] Población de clusters guardada en: {full_path}  ({pop_size} soluciones × {n} genes)")
    return str(full_path)

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: XBEB SOBRE UNA SOLUCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
 
def xie_beni_eb(
    deb_matrix: np.ndarray,
    medoids: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calcula el índice Xie-Beni de expresión (XBEB) para una solución individual.
 
    Aplica la fórmula del paper sobre la matriz de distancia de expresión (DEB),
    usando los medoides y etiquetas de la solución evaluada. La operación es
    completamente vectorizada para GPU y CPU.
 
    Parámetros
    ----------
    deb_matrix : np.ndarray — matriz (n×n) de distancias de expresión génica (DEB).
    medoids    : np.ndarray — arreglo 1-D de k índices de medoides (0-based),
                              corresponde a una fila de la población.
    labels     : np.ndarray — arreglo 1-D de n etiquetas de cluster (1-based),
                              salida de build_clusters() en medoids_clustering.py.
 
    Retorna
    -------
    float — valor del índice XBEB. Valores menores indican mejor solución.
            Retorna np.inf si dos medoides son idénticos (distancia 0 entre ellos),
            penalizando soluciones degeneradas.
 
    Ejemplo
    -------
    >>> xb_score = xie_beni_eb(deb_matrix, medoids, labels)
    >>> print(f"XBEB = {xb_score:.6f}")
    """
    xp  = cp if _GPU else np
 
    D   = xp.asarray(deb_matrix)   # (n, n)
    med = xp.asarray(medoids)      # (k,)
    lbl = xp.asarray(labels)       # (n,) — 1-based
    n   = D.shape[0]
    k   = med.shape[0]
 
    # ── Numerador: Σ_k Σ_{x_i ∈ C_k} D²(z_k, x_i) ──────────────────────────
    # Convertir etiquetas a 0-based para indexar la población de medoides.
    # Para cada elemento i se obtiene el índice del medoide de su cluster
    # y se extrae la distancia correspondiente de la matriz DEB en una sola
    # operación sin bucles.
    cluster_idx       = lbl - 1                        # (n,) 0-based
    medoid_per_elem   = med[cluster_idx]               # índice del medoide de cada elemento: (n,)
    row_idx           = xp.arange(n)
    dist_to_medoid    = D[row_idx, medoid_per_elem]    # D(z_k, x_i) por elemento: (n,)
    numerator         = xp.sum(dist_to_medoid ** 2)   # escalar
 
    # ── Denominador: n · min_{k ≠ l} D²(z_k, z_l) ───────────────────────────
    # Submatriz (k×k) con distancias entre todos los pares de medoides.
    # La diagonal se enmascara con inf para ignorar D(z_k, z_k) = 0.
    D_med      = D[xp.ix_(med, med)]                  # (k, k)
    diag_mask  = xp.eye(k, dtype=bool)
    D_med_off  = xp.where(diag_mask, xp.inf, D_med)   # diagonal → inf
    min_inter  = xp.min(D_med_off ** 2)               # min D²(z_k, z_l), k ≠ l
 
    # Pasar a Python float (necesario tanto en GPU como CPU)
    if _GPU:
        numerator  = float(cp.asnumpy(numerator))
        min_inter  = float(cp.asnumpy(min_inter))
    else:
        numerator  = float(numerator)
        min_inter  = float(min_inter)
 
    # Protección ante medoides duplicados (solución degenerada)
    if min_inter == 0.0:
        return np.inf
 
    return numerator / (n * min_inter)

def save_xb_score(
    individuo: str,
    xb_deb: float,
    xb_dbb: float,
    output_dir: str,
    filename: str = "xb_score.csv",
) -> str:
    """
    Guarda los scores XBEB y XBBB de una solución individual como .csv.

    Columnas del archivo
    --------------------
    individuo : str   — identificador de la solución evaluada.
    XBEB      : float — valor del índice Xie-Beni usando distancia de expresión (DEB).
    XBBB      : float — valor del índice Xie-Beni usando distancia de binding (DBB).

    Parámetros
    ----------
    individuo  : str   — nombre de la solución (ej. "solucion1").
    xb_deb     : float — valor retornado por xie_beni_eb() con DEB.
    xb_dbb     : float — valor retornado por xie_beni_eb() con DBB.
    output_dir : str   — carpeta destino (se crea si no existe).
    filename   : str   — nombre del archivo de salida.

    Retorna
    -------
    str — ruta completa del archivo guardado.

    Ejemplo
    -------
    >>> path = save_xb_score("solucion1", 0.127, 0.089, output_dir="resultados")
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "individuo": [individuo],
        "XBEB":      [xb_deb],
        "XBBB":      [xb_dbb],
    })

    full_path = out_path / filename
    df.to_csv(full_path, index=False)
    print(f"[save] Scores XB guardados en: {full_path}  (XBEB={xb_deb:.6f}, XBBB={xb_dbb:.6f})")
    return str(full_path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE COMPLETO (función de alto nivel)
# ══════════════════════════════════════════════════════════════════════════════

def run_medoid_clustering(
    deb_path: str,
    dbb_path: str,
    k: int,
    pop_size: int,
    output_dir: str,
    population_indices: list[list[int]] | None = None,
    distance_type: str = "DEB",
    seed: int | None = None,
) -> dict:
    """
    Pipeline completo: carga matrices → define población → construye clusters → guarda.

    Los nombres de genes se extraen automáticamente del índice de filas
    de la matriz .csv; no es necesario proporcionarlos por separado.

    Parámetros
    ----------
    deb_path           : str                  — ruta a la matriz DEB (.csv).
    dbb_path           : str                  — ruta a la matriz DBB (.csv).
    k                  : int                  — número de clusters.
    pop_size           : int                  — número de individuos en la población.
    output_dir         : str                  — carpeta de salida.
    population_indices : list[list[int]]      — si se proporciona, usa esta población
                                                (lista de listas de índices de medoides);
                                                si es None, genera aleatoriamente.
    distance_type      : "DEB"|"DBB"          — qué matriz usar para la asignación.
    seed               : int                  — semilla aleatoria (solo si population=None).

    Retorna
    -------
    dict con claves:
        "population"  : np.ndarray   — matriz (pop_size×k) de medoides usados.
        "all_labels"  : np.ndarray   — matriz (pop_size×n) de etiquetas de cluster.
        "gene_names"  : list[str]    — nombres de genes en orden.
        "output"      : str          — ruta del archivo .csv guardado.
        "dataframe"   : pd.DataFrame — el DataFrame resultante.

    Ejemplo
    -------
    >>> result = run_medoid_clustering(
    ...     deb_path="data/DEB.csv",
    ...     dbb_path="data/DBB.csv",
    ...     k=4,
    ...     pop_size=25,
    ...     output_dir="resultados/run1",
    ...     distance_type="DEB",
    ...     seed=42,
    ... )
    >>> print(result["dataframe"].head())
    """
    # 1. Cargar matrices — los nombres de genes vienen de la misma matriz
    deb, gene_names_deb = load_distance_matrix(deb_path)
    dbb, gene_names_dbb = load_distance_matrix(dbb_path)

    # Verificar coherencia entre matrices
    if deb.shape != dbb.shape:
        raise ValueError(
            f"Las matrices DEB {deb.shape} y DBB {dbb.shape} deben tener el mismo tamaño."
        )
    if gene_names_deb != gene_names_dbb:
        raise ValueError(
            "Los nombres de genes en DEB y DBB no coinciden. "
            "Verifica que ambas matrices correspondan al mismo dataset."
        )

    gene_names = gene_names_deb
    n = deb.shape[0]

    # 2. Validar distance_type
    distance_type = distance_type.upper()
    if distance_type not in ("DEB", "DBB"):
        raise ValueError("distance_type debe ser 'DEB' o 'DBB'.")

    # 3. Definir población de medoides
    if population_indices is not None:
        population = np.stack(
            [medoids_from_list(idx, n=n) for idx in population_indices]
        ).astype(np.int32)
        print(f"[población] Usando población proporcionada: {population.shape[0]} individuos, k={k}.")
    else:
        population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
        print(f"[población] Población aleatoria generada: {population.shape[0]} individuos, k={k}.")

    # Log de medoides por individuo
    for i, medoids in enumerate(population):
        genes_med = [gene_names[idx] for idx in medoids]
        print(f"  Individuo {i + 1:>3}: medoides → {genes_med}")

    # 4. Construir clusters para ambas matrices
    print(f"\n[clusters] Asignando {n} genes para {population.shape[0]} individuos ...")
    all_labels_deb = build_clusters_population(deb, population)
    all_labels_dbb = build_clusters_population(dbb, population)

    # Matriz activa para el CSV de clusters (según distance_type)
    all_labels = all_labels_deb if distance_type == "DEB" else all_labels_dbb

    # Resumen de distribución por individuo
    for i, row_labels in enumerate(all_labels):
        unique, counts = np.unique(row_labels, return_counts=True)
        dist = {int(c): int(cnt) for c, cnt in zip(unique, counts)}
        print(f"  Individuo {i + 1:>3}: distribución de clusters ({distance_type}) → {dist}")

    # 5. Guardar CSV de clusters
    fname = f"clusters_pop{population.shape[0]}_k{k}_{distance_type.lower()}.csv"
    clusters_path = save_clusters(
        all_labels=all_labels,
        gene_names=gene_names,
        output_dir=output_dir,
        filename=fname,
    )

    # 6. Calcular función objetivo con ambas matrices (individuo 0)
    xb_deb = xie_beni_eb(deb, population[0], all_labels_deb[0])
    xb_dbb = xie_beni_eb(dbb, population[0], all_labels_dbb[0])

    print(f"  XBEB = {xb_deb:.6f}" + ("  (solución degenerada)" if xb_deb == np.inf else ""))
    print(f"  XBBB = {xb_dbb:.6f}" + ("  (solución degenerada)" if xb_dbb == np.inf else ""))

    scores_path = save_xb_score(
        individuo="solucion1",
        xb_deb=xb_deb,
        xb_dbb=xb_dbb,
        output_dir=output_dir,
        filename="resultado.csv",
    )

    df = pd.read_csv(clusters_path, index_col=0)
    return {
        "population":     population,
        "all_labels":     all_labels,
        "all_labels_deb": all_labels_deb,
        "all_labels_dbb": all_labels_dbb,
        "gene_names":     gene_names,
        "output":         clusters_path,
        "scores_output":  scores_path,
        "dataframe":      df,
        "xb_deb":         xb_deb,
        "xb_dbb":         xb_dbb,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO (se ejecuta solo si llamas el script directamente)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    deb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBE.csv"
    dbb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBI.csv"
    k          = 4
    pop_size   = 25
    output_dir = r"C:\Users\benja\Desktop\workspace\Thesis\Results"
    dist_type  = "DEB"
    seed       = 42

    result = run_medoid_clustering(
        deb_path=deb_path,
        dbb_path=dbb_path,
        k=k,
        pop_size=pop_size,
        output_dir=output_dir,
        distance_type=dist_type,
        seed=seed,
    )

    print("\n── Primeras filas del resultado ──")
    print(result["dataframe"].head(10).to_string())