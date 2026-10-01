"""
Operadores de interacción entre soluciones: (k-1)-point crossover,
controller-random mutation y selección por torneo binario (NSGA-II).
"""

############################
# Importaciones
############################
import numpy as np              # Operaciones matemáticas.

##################################
# Conjunto de funciones.
##################################

def _repair_duplicates(individual: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Repara un cromosoma reemplazando valores duplicados por índices válidos no usados.

    Recorre el cromosoma de izquierda a derecha; la primera ocurrencia de cada valor 
    se mantiene, las ocurrencias repetidas se sustituyen por un índice aleatorio en 
    [0, n-1] que no esté ya presente en el cromosoma.
    """
    repaired = individual.copy()                            # Copia del individuo para modificación.
    seen = set()                                            # Arreglo de indices de genes vistos tanto en la solución como en la reparación.
    used = set(individual.tolist())                         # Arreglo de indices de genes ya usados en reparación.

    # Genes disponibles puestos en un conjunto aleatorio.
    available = [i for i in range(n) if i not in used]
    rng.shuffle(available)

    # Reparación.
    for i, val in enumerate(repaired):
        if val in seen:
            new_val = available.pop()
            seen.add(new_val)
            used.add(new_val)
            repaired[i] = new_val
        else:
            seen.add(val)

    return repaired

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
    n              : int        — número total de elementos del dataset (usado para reparación de duplicados).
    crossover_prob : float      — probabilidad de aplicar el crossover (default 0.80).
    rng            : np.random.Generator — generador aleatorio (opcional). Si es None, se crea uno nuevo sin semilla fija.
    """
    # Definición del administrador pseudo-aleatorio.
    if rng is None:
        rng = np.random.default_rng()

    # Cantidad de medoides.
    k = len(parent1)

    # Caso 1 - Sin crossover: los hijos son copias de los padres.
    if rng.random() > crossover_prob:
        return parent1.copy(), parent2.copy()
    # Caso 2 - 1 medoide: No se piede hacer intercambios.
    if k == 1:
        return parent1.copy(), parent2.copy()

    # Seleccionar k-1 puntos de corte únicos en [1, k-1] y ordenarlos.
    n_cuts = k - 1
    cut_points = sorted(rng.choice(np.arange(1, k), size=n_cuts, replace=False))
    
    # Construir segmentos alternados: [0:c1), [c1:c2), ..., [c_{k-1}:k)
    boundaries = [0] + list(cut_points) + [k]

    # Construcción de hijos vacios.
    child1 = np.empty(k, dtype=parent1.dtype)
    child2 = np.empty(k, dtype=parent2.dtype)

    # fusión entre hijos.
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

def controller_random_mutation(
    individual: np.ndarray,
    n: int,
    mutation_prob: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Aplica el operador controller-random mutation sobre un individuo.

    Con probabilidad mutation_prob, se selecciona una única posición al
    azar del cromosoma y se reemplaza el medoide en esa posición por un
    elemento del dataset que no esté presente en el cromosoma (manteniendo
    la unicidad de medoides).

    Parámetros
    ----------
    individual    : np.ndarray — cromosoma a mutar, shape (k,).
    n             : int        — número total de elementos del dataset.
    mutation_prob : float      — probabilidad de que ocurra la mutación sobre el individuo (default 0.01).
    rng           : np.random.Generator — generador aleatorio (opcional). Si es None, se crea uno nuevo sin semilla fija.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = len(individual)
    mutated = individual.copy()

    # Si k == n no quedan elementos disponibles para mutar.
    if k >= n:
        return mutated

    # Control: la mutación ocurre (o no) una sola vez por individuo.
    if rng.random() < mutation_prob:
        pos = rng.integers(k)                          # posición seleccionada al azar
        used = set(mutated.tolist())
        available = [i for i in range(n) if i not in used]
        if available:
            mutated[pos] = rng.choice(available)

    return mutated

def binary_tournament_selection(
    population: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    n_offspring: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Selecciona padres mediante torneo binario, usando ranking de Pareto y,
    en caso de empate, crowding distance (NSGA-II).

    Parámetros
    ----------
    population  : np.ndarray — población actual, shape (pop_size, k).
    ranks       : np.ndarray — rank (nivel de no-dominancia) de cada individuo.
    crowding    : np.ndarray — crowding distance de cada individuo.
    n_offspring : int        — número de padres a seleccionar.
    rng         : np.random.Generator — generador aleatorio.

    Retorna
    -------
    np.ndarray — población de padres seleccionados, shape (n_offspring, k).

    Nota
    ----
    Esta función NO llama a la función objetivo: opera sobre ranks y crowding
    ya calculados, por lo que no consume presupuesto de evaluaciones.
    """
    pop_size = population.shape[0]
    selected = np.empty((n_offspring, population.shape[1]), dtype=population.dtype)

    for i in range(n_offspring):
        a, b = rng.integers(0, pop_size), rng.integers(0, pop_size)
        # Gana el de mejor (menor) rank; si empatan, mayor crowding distance.
        if ranks[a] < ranks[b]:
            winner = a
        elif ranks[b] < ranks[a]:
            winner = b
        else:
            winner = a if crowding[a] >= crowding[b] else b
        selected[i] = population[winner]

    return selected