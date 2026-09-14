"""
Ordenamiento por dominancia de Pareto (non-dominated sort) y crowding
distance para 2 objetivos (XB_GE, XB_BI).
"""

import numpy as np


def _dominates(obj_a, obj_b) -> bool:
    """A domina a B (minimización) si A no es peor en ningún objetivo y es mejor en al menos uno."""
    not_worse = all(a <= b for a, b in zip(obj_a, obj_b))
    strictly_better = any(a < b for a, b in zip(obj_a, obj_b))
    return not_worse and strictly_better


def non_dominated_sort(objectives) -> list[list[int]]:
    """Ordena una población en frentes de Pareto (F1, F2, ...) según dominancia."""
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