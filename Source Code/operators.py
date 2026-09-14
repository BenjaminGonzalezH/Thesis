"""
Operadores de interacción entre soluciones: (k-1)-point crossover,
controller-random mutation y selección por torneo binario (NSGA-II).
"""

import numpy as np


def _repair_duplicates(individual: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Repara un cromosoma reemplazando valores duplicados por índices válidos no usados."""
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


def k_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n: int,
    crossover_prob: float = 0.80,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica (k-1)-point crossover sobre un par de padres. Con probabilidad
    (1 - crossover_prob), no hay cruce y los hijos son copias de los padres.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = len(parent1)

    if rng.random() > crossover_prob:
        return parent1.copy(), parent2.copy()
    if k == 1:
        return parent1.copy(), parent2.copy()

    n_cuts = k - 1
    cut_points = sorted(rng.choice(np.arange(1, k), size=n_cuts, replace=False))
    boundaries = [0] + list(cut_points) + [k]

    child1 = np.empty(k, dtype=parent1.dtype)
    child2 = np.empty(k, dtype=parent2.dtype)

    for seg_idx in range(len(boundaries) - 1):
        start, end = boundaries[seg_idx], boundaries[seg_idx + 1]
        if seg_idx % 2 == 0:
            child1[start:end] = parent1[start:end]
            child2[start:end] = parent2[start:end]
        else:
            child1[start:end] = parent2[start:end]
            child2[start:end] = parent1[start:end]

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
    Con probabilidad mutation_prob, selecciona una única posición al azar
    del cromosoma y la reemplaza por un elemento no presente en él.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = len(individual)
    mutated = individual.copy()

    if k >= n:
        return mutated

    if rng.random() < mutation_prob:
        pos = rng.integers(k)
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
    Selecciona padres por torneo binario, priorizando menor rank y,
    en caso de empate, mayor crowding distance. No consume presupuesto
    de evaluaciones (opera sobre ranks/crowding ya calculados).
    """
    pop_size = population.shape[0]
    selected = np.empty((n_offspring, population.shape[1]), dtype=population.dtype)

    for i in range(n_offspring):
        a, b = rng.integers(0, pop_size), rng.integers(0, pop_size)
        if ranks[a] < ranks[b]:
            winner = a
        elif ranks[b] < ranks[a]:
            winner = b
        else:
            winner = a if crowding[a] >= crowding[b] else b
        selected[i] = population[winner]

    return selected