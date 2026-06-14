"""
crossover_mutation.py
----------------------
Operadores genéticos de cruce (crossover) y mutación para el algoritmo
NSGA-II en MOC-GaPBK (Parraga-Alava et al., 2018), aplicados sobre arreglos
de medoides (cromosomas).

Operadores implementados:

    1. (k-1)-point crossover
       Se seleccionan k-1 puntos de corte en ambos padres. Los medoides
       entre dichos puntos se intercambian entre los dos individuos,
       generando dos hijos.

    2. Controller-random mutation
       Para cada posición del cromosoma, con probabilidad `mutation_prob`,
       se reemplaza el medoide actual por un elemento del dataset que no
       esté presente en el cromosoma.

Restricción de validez
-----------------------
Cada cromosoma es un arreglo de k índices ÚNICOS en [0, n-1] (sin
repetición), ya que cada posición representa el medoide de un cluster
distinto. Tanto el crossover como la mutación pueden introducir
duplicados; ambos operadores incluyen un mecanismo de reparación que
sustituye los duplicados por índices válidos no usados.

Nota sobre GPU
---------------
Los cromosomas tienen longitud k (típicamente 4-6), por lo que no se
justifica vectorización GPU: el overhead de transferencia superaría
cualquier beneficio. Estas operaciones se mantienen en NumPy puro.
"""

##################################
# Imports
##################################
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ══════════════════════════════════════════════════════════════════════════════

def _validate_chromosome(individual: np.ndarray, n: int, name: str = "individual") -> None:
    """
    Valida que un cromosoma sea un arreglo 1-D de índices únicos en [0, n-1].

    Parámetros
    ----------
    individual : np.ndarray — arreglo de medoides a validar.
    n          : int        — número total de elementos del dataset.
    name       : str        — nombre del cromosoma, usado en mensajes de error.

    Lanza
    -----
    ValueError si el cromosoma no es 1-D, tiene duplicados o índices fuera de rango.
    """
    if individual.ndim != 1:
        raise ValueError(f"'{name}' debe ser un arreglo 1-D, se recibió shape={individual.shape}.")
    if len(set(individual.tolist())) != len(individual):
        raise ValueError(f"'{name}' contiene medoides duplicados: {individual.tolist()}.")
    if individual.min() < 0 or individual.max() >= n:
        raise ValueError(f"'{name}' tiene índices fuera de rango [0, {n-1}]: {individual.tolist()}.")


def _repair_duplicates(individual: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Repara un cromosoma reemplazando valores duplicados por índices válidos no usados.

    Recorre el cromosoma de izquierda a derecha; la primera ocurrencia de cada
    valor se mantiene, las ocurrencias repetidas se sustituyen por un índice
    aleatorio en [0, n-1] que no esté ya presente en el cromosoma.

    Parámetros
    ----------
    individual : np.ndarray — cromosoma potencialmente con duplicados.
    n          : int        — número total de elementos del dataset.
    rng        : np.random.Generator — generador aleatorio.

    Retorna
    -------
    np.ndarray — cromosoma reparado, con valores únicos en [0, n-1].

    Ejemplo
    -------
    >>> rng = np.random.default_rng(0)
    >>> _repair_duplicates(np.array([3, 3, 7]), n=10, rng=rng)
    array([3, 8, 7])  # el 3 repetido se reemplaza por un valor no usado
    """
    repaired = individual.copy()
    seen = set()
    used = set(individual.tolist())

    available = [i for i in range(n) if i not in used]
    rng.shuffle(available)

    for i, val in enumerate(repaired):
        if val in seen:
            new_val = available.pop()
            seen.add(new_val)
            used.add(new_val)
            repaired[i] = new_val
        else:
            seen.add(val)

    return repaired


# ══════════════════════════════════════════════════════════════════════════════
# 1. (K-1)-POINT CROSSOVER
# ══════════════════════════════════════════════════════════════════════════════

def k_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n: int,
    crossover_prob: float = 0.80,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica el operador (k-1)-point crossover sobre un par de padres.

    Se seleccionan k-1 puntos de corte aleatorios (posiciones internas del
    cromosoma), dividiendo a ambos padres en segmentos alternados. Los
    segmentos se intercambian entre los dos padres para formar dos hijos.
    Si el crossover introduce medoides duplicados en alguno de los hijos,
    se aplica reparación automática.

    Con probabilidad (1 - crossover_prob), no se realiza cruce y los hijos
    son copias exactas de los padres.

    Parámetros
    ----------
    parent1        : np.ndarray — cromosoma del primer padre, shape (k,).
    parent2        : np.ndarray — cromosoma del segundo padre, shape (k,).
    n              : int        — número total de elementos del dataset
                                  (usado para reparación de duplicados).
    crossover_prob : float      — probabilidad de aplicar el crossover (default 0.80).
    rng            : np.random.Generator — generador aleatorio (opcional).
                                  Si es None, se crea uno nuevo sin semilla fija.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray] — (hijo1, hijo2), ambos shape (k,) con
    medoides únicos en [0, n-1].

    Ejemplo
    -------
    >>> rng = np.random.default_rng(42)
    >>> p1 = np.array([10, 25, 3, 47])
    >>> p2 = np.array([5, 19, 33, 8])
    >>> child1, child2 = k_point_crossover(p1, p2, n=100, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    _validate_chromosome(parent1, n, "parent1")
    _validate_chromosome(parent2, n, "parent2")
    if parent1.shape != parent2.shape:
        raise ValueError(
            f"parent1 y parent2 deben tener la misma longitud: "
            f"{parent1.shape} vs {parent2.shape}."
        )

    k = len(parent1)

    # Sin crossover: los hijos son copias de los padres.
    if rng.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    # k=1 → no hay puntos de corte posibles, no se puede cruzar.
    if k == 1:
        return parent1.copy(), parent2.copy()

    # Seleccionar k-1 puntos de corte únicos en [1, k-1] y ordenarlos.
    n_cuts = min(k - 1, k - 1)  # número de puntos de corte = k-1
    cut_points = sorted(rng.choice(np.arange(1, k), size=n_cuts, replace=False))

    # Construir segmentos alternados: [0:c1), [c1:c2), ..., [c_{k-1}:k)
    boundaries = [0] + list(cut_points) + [k]

    child1 = np.empty(k, dtype=parent1.dtype)
    child2 = np.empty(k, dtype=parent2.dtype)

    for seg_idx in range(len(boundaries) - 1):
        start, end = boundaries[seg_idx], boundaries[seg_idx + 1]
        if seg_idx % 2 == 0:
            # Segmento par: hijo1 toma de padre1, hijo2 toma de padre2.
            child1[start:end] = parent1[start:end]
            child2[start:end] = parent2[start:end]
        else:
            # Segmento impar: se intercambian los segmentos.
            child1[start:end] = parent2[start:end]
            child2[start:end] = parent1[start:end]

    # Reparar posibles duplicados generados por el intercambio.
    child1 = _repair_duplicates(child1, n, rng)
    child2 = _repair_duplicates(child2, n, rng)

    return child1, child2


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTROLLER-RANDOM MUTATION
# ══════════════════════════════════════════════════════════════════════════════

def controller_random_mutation(
    individual: np.ndarray,
    n: int,
    mutation_prob: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Aplica el operador controller-random mutation sobre un individuo.

    Para cada posición del cromosoma, con probabilidad `mutation_prob`,
    el medoide actual se reemplaza por un elemento del dataset que no esté
    presente en el cromosoma (manteniendo la unicidad de medoides).

    Parámetros
    ----------
    individual    : np.ndarray — cromosoma a mutar, shape (k,).
    n             : int        — número total de elementos del dataset.
    mutation_prob : float      — probabilidad de mutación por posición (default 0.01).
    rng           : np.random.Generator — generador aleatorio (opcional).
                                 Si es None, se crea uno nuevo sin semilla fija.

    Retorna
    -------
    np.ndarray — cromosoma mutado, shape (k,), con medoides únicos en [0, n-1].
                 Si k == n, no hay elementos disponibles para mutar y se
                 retorna una copia sin cambios.

    Ejemplo
    -------
    >>> rng = np.random.default_rng(42)
    >>> ind = np.array([10, 25, 3, 47])
    >>> mutated = controller_random_mutation(ind, n=100, mutation_prob=0.01, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    _validate_chromosome(individual, n, "individual")

    k = len(individual)
    mutated = individual.copy()
    used = set(mutated.tolist())

    # Si k == n no quedan elementos disponibles para mutar.
    if k >= n:
        return mutated

    for pos in range(k):
        if rng.random() < mutation_prob:
            available = [i for i in range(n) if i not in used]
            if not available:
                break  # no quedan elementos disponibles
            new_val = rng.choice(available)
            used.discard(mutated[pos])
            used.add(new_val)
            mutated[pos] = new_val

    return mutated


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE: CROSSOVER + MUTACIÓN SOBRE UN PAR DE PADRES
# ══════════════════════════════════════════════════════════════════════════════

def apply_operators(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n: int,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica secuencialmente crossover y mutación sobre un par de padres,
    generando dos hijos listos para integrar la siguiente generación.

    Parámetros
    ----------
    parent1        : np.ndarray — cromosoma del primer padre, shape (k,).
    parent2        : np.ndarray — cromosoma del segundo padre, shape (k,).
    n              : int        — número total de elementos del dataset.
    crossover_prob : float      — probabilidad de aplicar crossover (default 0.80).
    mutation_prob  : float      — probabilidad de mutación por posición (default 0.01).
    seed           : int        — semilla para reproducibilidad (opcional).

    Retorna
    -------
    tuple[np.ndarray, np.ndarray] — (hijo1, hijo2) tras crossover + mutación.

    Ejemplo
    -------
    >>> p1 = np.array([10, 25, 3, 47])
    >>> p2 = np.array([5, 19, 33, 8])
    >>> child1, child2 = apply_operators(p1, p2, n=100, seed=42)
    >>> print(child1, child2)
    """
    rng = np.random.default_rng(seed)

    child1, child2 = k_point_crossover(
        parent1, parent2, n=n, crossover_prob=crossover_prob, rng=rng
    )
    child1 = controller_random_mutation(child1, n=n, mutation_prob=mutation_prob, rng=rng)
    child2 = controller_random_mutation(child2, n=n, mutation_prob=mutation_prob, rng=rng)

    return child1, child2


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE: CROSSOVER + MUTACIÓN SOBRE LA POBLACIÓN COMPLETA
# ══════════════════════════════════════════════════════════════════════════════

def apply_operators_population(
    population: np.ndarray,
    n: int,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """
    Aplica crossover + mutación sobre una población completa, generando una
    población de hijos del mismo tamaño que la de entrada.

    Estrategia de emparejamiento
    -----------------------------
    1. Los índices de la población se mezclan aleatoriamente (shuffle).
    2. Se forman pares consecutivos de individuos mezclados: (0,1), (2,3), ...
    3. Cada par produce 2 hijos mediante apply_operators().
    4. Si pop_size es impar, el último individuo sin pareja se cruza con un
       individuo aleatorio de la población (elegido con reemplazo) y se
       conserva solo uno de los dos hijos generados, de modo que la
       población resultante tenga exactamente pop_size individuos.

    Parámetros
    ----------
    population     : np.ndarray — población de padres, shape (pop_size, k).
    n              : int        — número total de elementos del dataset.
    crossover_prob : float      — probabilidad de aplicar crossover (default 0.80).
    mutation_prob  : float      — probabilidad de mutación por posición (default 0.01).
    seed           : int        — semilla para reproducibilidad (opcional).

    Retorna
    -------
    np.ndarray — población de hijos, shape (pop_size, k), cada fila con
    medoides únicos en [0, n-1].

    Ejemplo
    -------
    >>> from medoids_clustering import random_medoids_pop
    >>> population = random_medoids_pop(n=100, k=4, pop_size=10, seed=0)
    >>> offspring = apply_operators_population(population, n=100, seed=42)
    >>> print(offspring.shape)  # (10, 4)
    """
    if population.ndim != 2:
        raise ValueError(f"'population' debe ser 2-D (pop_size, k), se recibió shape={population.shape}.")

    pop_size, k = population.shape
    rng = np.random.default_rng(seed)

    # 1. Mezclar índices de la población.
    shuffled_idx = rng.permutation(pop_size)

    offspring = np.empty((pop_size, k), dtype=population.dtype)
    n_filled = 0

    # 2-3. Emparejar consecutivos y generar hijos.
    n_pairs = pop_size // 2
    for p in range(n_pairs):
        idx1, idx2 = shuffled_idx[2 * p], shuffled_idx[2 * p + 1]
        parent1, parent2 = population[idx1], population[idx2]

        child1, child2 = k_point_crossover(
            parent1, parent2, n=n, crossover_prob=crossover_prob, rng=rng
        )
        child1 = controller_random_mutation(child1, n=n, mutation_prob=mutation_prob, rng=rng)
        child2 = controller_random_mutation(child2, n=n, mutation_prob=mutation_prob, rng=rng)

        offspring[n_filled] = child1
        offspring[n_filled + 1] = child2
        n_filled += 2

    # 4. Individuo impar sin pareja: cruzarlo con uno aleatorio y conservar un solo hijo.
    if pop_size % 2 == 1:
        leftover_idx = shuffled_idx[-1]
        partner_idx = rng.integers(0, pop_size)

        parent1, parent2 = population[leftover_idx], population[partner_idx]

        child1, _ = k_point_crossover(
            parent1, parent2, n=n, crossover_prob=crossover_prob, rng=rng
        )
        child1 = controller_random_mutation(child1, n=n, mutation_prob=mutation_prob, rng=rng)

        offspring[n_filled] = child1
        n_filled += 1

    return offspring


# ══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from solution_definition import load_distance_matrix, random_medoids_pop

    deb_path = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBE.csv"
    k        = 4
    pop_size = 9   # impar, para ejercitar el caso del individuo sin pareja
    seed     = 42

    # 1. Cargar matriz para obtener n
    deb, gene_names = load_distance_matrix(deb_path)
    n = len(gene_names)

    # ── Ejemplo A: par de individuos ────────────────────────────────────────
    population = random_medoids_pop(n=n, k=k, pop_size=2, seed=seed)
    parent1, parent2 = population[0], population[1]

    print("── Ejemplo A: par de individuos ──")
    print(f"Padre 1: {parent1} → {[gene_names[i] for i in parent1]}")
    print(f"Padre 2: {parent2} → {[gene_names[i] for i in parent2]}")

    child1, child2 = apply_operators(
        parent1, parent2, n=n,
        crossover_prob=0.80, mutation_prob=0.01,
        seed=seed,
    )

    print(f"Hijo 1:  {child1} → {[gene_names[i] for i in child1]}")
    print(f"Hijo 2:  {child2} → {[gene_names[i] for i in child2]}")

    # ── Ejemplo B: población completa ───────────────────────────────────────
    print(f"\n── Ejemplo B: población completa (pop_size={pop_size}) ──")
    parent_population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    offspring_population = apply_operators_population(
        parent_population, n=n,
        crossover_prob=0.80, mutation_prob=0.01,
        seed=seed,
    )

    print(f"Población de padres  shape: {parent_population.shape}")
    print(f"Población de hijos   shape: {offspring_population.shape}")
    for i in range(pop_size):
        print(f"  Padre {i}: {parent_population[i]}  →  Hijo {i}: {offspring_population[i]}")