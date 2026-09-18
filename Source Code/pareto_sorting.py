"""
Ordenamiento por dominancia de Pareto (non-dominated sort) y crowding
distance para 2 objetivos (XB_GE, XB_BI).
"""

import numpy as np


def dominance_matrix(objectives: np.ndarray) -> np.ndarray:
    obj = np.asarray(objectives)
    le = obj[:, None, :] <= obj[None, :, :]
    lt = obj[:, None, :] <  obj[None, :, :]
    return np.all(le, axis=2) & np.any(lt, axis=2)   # dom[p, q] = True si p domina a q

def non_dominated_sort(objectives) -> list[list[int]]:
    """
    Ordena una población en frentes de Pareto (F1, F2, ...) según dominancia,
    replicando el procedimiento de NSGA-II. La comparación de dominancia se
    vectoriza con dominance_matrix (O(n²) en NumPy) en vez de un doble loop
    de Python con _dominates.

    Parámetros
    ----------
    objectives : np.ndarray de shape (pop_size, 2), o lista de tuplas
                 (XBEB, XBBB) — un par de objetivos por individuo.
    """
    objectives = np.asarray(objectives)
    n = len(objectives)

    dom = dominance_matrix(objectives)                  # dom[p, q] = True si p domina a q
    domination_count = dom.sum(axis=0).tolist()         # cuántos dominan a cada solución p
    dominated_by = [np.where(dom[p])[0].tolist() for p in range(n)]  # a quiénes domina p

    fronts: list[list[int]] = [[p for p in range(n) if domination_count[p] == 0]]

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

    return fronts[:-1] 

def crowding_distance(objectives, front: list[int]) -> dict[int, float]:
    """Crowding distance de las soluciones de un frente, para 2 objetivos."""
    distances = {idx: 0.0 for idx in front}
    size = len(front)

    if size <= 2:
        return {idx: np.inf for idx in front}

    for obj_idx in range(2):
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