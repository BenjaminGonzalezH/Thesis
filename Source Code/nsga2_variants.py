"""
Variantes de run_nsga2 desarrolladas en Notes_2-LSOperators.ipynb, que
agregan cada uno de los operadores de búsqueda local de LS.py como
estrategia de intensificación/diversificación aplicada sobre el frente no
dominado F1 de Rv = Pv ∪ Qv, antes de la selección de sobrevivientes.

- run_nsga2_MOPR  : agrega Multi-Objective Path-Relinking (intensificación).
- run_nsga2_PLS   : agrega Pareto Local Search (diversificación).
- run_nsga2_MOLS  : agrega L-MOLS o N-MOLS, según `mode`.

El resto del ciclo (selección por torneo, crossover, mutación, truncamiento
por rank+crowding) es IDÉNTICO a run_nsga2 (nsga2.py). Todas comparten la
MISMA cuota global (solution_encoding.OBJ_FUNCTION_CALLS vs
max_obj_function_calls) con los operadores LS, por lo que nunca se excede
max_obj_function_calls.
"""

import numpy as np

import solution_encoding
from solution_encoding import random_medoids_pop, build_clusters_population
from pareto_sorting import non_dominated_sort
from operators import binary_tournament_selection, k_point_crossover, controller_random_mutation
from performance import hypervolume_from_origin
from nsga2 import evaluate_population, compute_ranks_and_crowding, _ensure_unique
from LS import (
    multi_objective_path_relinking,
    pareto_local_search,
    build_neighborhood_matrix,
    l_mols,
    n_mols,
)


#######################################################
# run_nsga2_MOPR: variante de run_nsga2 que agrega Multi-Objective
# Path-Relinking (MOPR) como estrategia de intensificación
#######################################################

def run_nsga2_MOPR(
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Variante de run_nsga2 que agrega MOPR como estrategia de intensificación
    (líneas 11-12 del Algoritmo 1, Parraga-Alava et al. 2018): tras construir
    Rv = Pv ∪ Qv, se extrae su frente no dominado F1, se aplica MOPR sobre
    pares consecutivos de F1, y las soluciones NUEVAS que aporta (evaluadas
    respetando la misma cuota global) se agregan de vuelta a Rv antes de la
    selección de sobrevivientes. El resto del ciclo (selección por torneo,
    crossover, mutación, truncamiento por rank+crowding) es IDÉNTICO a
    run_nsga2.

    Criterio de paro EXACTO: MOPR comparte la MISMA cuota global
    (solution_encoding.OBJ_FUNCTION_CALLS vs max_obj_function_calls) que el
    resto del ciclo, por lo que puede detener una trayectoria a medio camino
    si el presupuesto se agota (nunca se excede max_obj_function_calls).
    """
    rng = np.random.default_rng(seed)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, deb_matrix, dbb_matrix, max_obj_function_calls=max_obj_function_calls,
    )

    initial_fronts = non_dominated_sort(objectives) if len(objectives) else [[]]
    initial_f1 = initial_fronts[0]

    history_objectives = [objectives.copy()]
    history_labels = [labels_pop.copy()]
    history_hv = [hypervolume_from_origin(objectives[:, 0], objectives[:, 1])] if len(objectives) else []
    history_hv_f1 = [hypervolume_from_origin(objectives[initial_f1, 0], objectives[initial_f1, 1])] if len(objectives) else []

    gen = 0
    while solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls and len(population) >= 2:
        gen += 1
        pop_size_actual = len(population)

        # ── Selección + offspring (idéntico a run_nsga2) ────────────────────
        ranks, crowding, _ = compute_ranks_and_crowding(objectives)
        parents = binary_tournament_selection(population, ranks, crowding, n_offspring=pop_size_actual, rng=rng)

        existing_sets = {frozenset(ind.tolist()) for ind in population}

        offspring_candidates = np.empty_like(parents)
        for i in range(0, pop_size_actual - 1, 2):
            c1, c2 = k_point_crossover(parents[i], parents[i + 1], n=n, crossover_prob=crossover_prob, rng=rng)
            c1 = controller_random_mutation(c1, n=n, mutation_prob=mutation_prob, rng=rng)
            c2 = controller_random_mutation(c2, n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[i] = _ensure_unique(c1, existing_sets, n, k, rng)
            offspring_candidates[i + 1] = _ensure_unique(c2, existing_sets, n, k, rng)
        if pop_size_actual % 2 == 1:
            c_last = controller_random_mutation(parents[-1].copy(), n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[-1] = _ensure_unique(c_last, existing_sets, n, k, rng)

        offspring, offspring_objectives, offspring_labels = evaluate_population(
            offspring_candidates, deb_matrix, dbb_matrix, max_obj_function_calls=max_obj_function_calls,
        )
        if len(offspring) == 0:
            break

        combined_population = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])
        combined_labels = np.vstack([labels_pop, offspring_labels])

        # ── MOPR: ÚNICO agregado respecto a run_nsga2 ───────────────────────
        # Intensificación sobre el frente no dominado de Rv = Pv ∪ Qv.
        if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls:
            rv_fronts = non_dominated_sort(combined_objectives)
            f1_idx = rv_fronts[0]
            f1_pop = combined_population[f1_idx]

            if len(f1_pop) >= 2:
                pooled = [f1_pop]
                for i in range(len(f1_pop) - 1):
                    if solution_encoding.OBJ_FUNCTION_CALLS >= max_obj_function_calls:
                        break
                    C1, C2 = f1_pop[i], f1_pop[i + 1]
                    if frozenset(C1.tolist()) == frozenset(C2.tolist()):
                        continue
                    mopr_result = multi_objective_path_relinking(
                        C1, C2, deb_matrix, dbb_matrix, max_obj_function_calls, verbose=False,
                    )
                    if len(mopr_result["f1"]) > 0:
                        pooled.append(mopr_result["f1"])

                mopr_pop = np.vstack(pooled)
                seen = set()
                unique_rows = []
                for sol in mopr_pop:
                    key = frozenset(sol.tolist())
                    if key not in seen:
                        seen.add(key)
                        unique_rows.append(sol)
                mopr_pop = np.stack(unique_rows)

                # Solo evaluar (y agregar) las soluciones REALMENTE nuevas
                # que MOPR aportó — evita recalcular objetivos ya conocidos.
                existing_keys = {frozenset(sol.tolist()) for sol in combined_population}
                new_rows = [sol for sol in mopr_pop if frozenset(sol.tolist()) not in existing_keys]
                if new_rows:
                    new_solutions = np.stack(new_rows)
                    new_solutions, new_objectives, new_labels = evaluate_population(
                        new_solutions, deb_matrix, dbb_matrix, max_obj_function_calls=max_obj_function_calls,
                    )
                    if len(new_solutions) > 0:
                        combined_population = np.vstack([combined_population, new_solutions])
                        combined_objectives = np.vstack([combined_objectives, new_objectives])
                        combined_labels = np.vstack([combined_labels, new_labels])

        # ── Selección de sobrevivientes (idéntica a run_nsga2) ──────────────
        _, combined_crowding, combined_fronts = compute_ranks_and_crowding(combined_objectives)

        target_size = min(pop_size, len(combined_population))
        next_indices: list[int] = []
        for front in combined_fronts:
            if len(next_indices) + len(front) <= target_size:
                next_indices.extend(front)
            else:
                remaining_slots = target_size - len(next_indices)
                front_sorted = sorted(front, key=lambda idx: combined_crowding[idx], reverse=True)
                next_indices.extend(front_sorted[:remaining_slots])
                break

        population = combined_population[next_indices]
        objectives = combined_objectives[next_indices]
        labels_pop = combined_labels[next_indices]

        current_fronts = non_dominated_sort(objectives)
        f1_indices = current_fronts[0]
        f1_hv = hypervolume_from_origin(objectives[f1_indices, 0], objectives[f1_indices, 1])

        history_objectives.append(objectives.copy())
        history_labels.append(labels_pop.copy())
        history_hv.append(hypervolume_from_origin(objectives[:, 0], objectives[:, 1]))
        history_hv_f1.append(f1_hv)

        mean_hv = np.mean(history_hv[-1])
        mean_hv_f1 = np.mean(f1_hv)
        print(
            f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  "
            f"población = {len(population)}  |  F1 = {len(f1_indices)}  |  "
            f"HV promedio = {mean_hv:.6f}  |  HV F1 promedio = {mean_hv_f1:.6f}"
        )

    print(f"\nCriterio de paro EXACTO: {solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_function_calls} llamadas a la función objetivo, {gen} generaciones completadas.")

    return {
        "population": population,
        "objectives": objectives,
        "labels": labels_pop,
        "fronts": non_dominated_sort(objectives),
        "history_objectives": history_objectives,
        "history_labels": history_labels,
        "history_hv": history_hv,
        "history_hv_mean": [float(np.mean(hv)) for hv in history_hv],
        "history_hv_f1": history_hv_f1,
        "history_hv_f1_mean": [float(np.mean(hv)) for hv in history_hv_f1],
        "generations_run": gen,
    }


#######################################################
# run_nsga2_PLS: variante de run_nsga2 que agrega Pareto Local Search
# (PLS) como estrategia de diversificación
#######################################################

def run_nsga2_PLS(
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    pls_local_budget: int | None = None,
    seed: int | None = None,
) -> dict:
    """
    Variante de run_nsga2 que agrega PLS como estrategia de diversificación
    (líneas 13-14 del Algoritmo 1, Parraga-Alava et al. 2018): tras construir
    Rv = Pv ∪ Qv, se extrae su frente no dominado F1 y se aplica PLS sobre
    él; las soluciones NUEVAS que aporta se agregan de vuelta a Rv antes de
    la selección de sobrevivientes. El resto del ciclo es IDÉNTICO a
    run_nsga2 — MOPR no se agrega en esta variante.

    Parámetros
    ----------
    pls_local_budget : int | None — cuota de evaluaciones que PLS puede
        consumir POR GENERACIÓN (ver pareto_local_search en LS.py). SIN
        este límite, PLS puede agotar TODO max_obj_function_calls en la
        primera generación, dejando 0 evaluaciones para el resto del
        NSGA-II (comprobado: con pop_size=20, max_obj_function_calls=3000,
        sin límite el run termina en 1 generación; con
        pls_local_budget=150 —5% del total— corre 18 generaciones
        completas). Se recomienda fijarlo como una fracción de
        max_obj_function_calls, análogo al parámetro `ls_budget` de
        Toledo (2021), Sección 3.3.6.

    Criterio de paro EXACTO: PLS comparte la MISMA cuota global
    (solution_encoding.OBJ_FUNCTION_CALLS vs max_obj_function_calls) que el
    resto del ciclo — nunca se excede max_obj_function_calls.
    """
    rng = np.random.default_rng(seed)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, deb_matrix, dbb_matrix, max_obj_function_calls=max_obj_function_calls,
    )

    initial_fronts = non_dominated_sort(objectives) if len(objectives) else [[]]
    initial_f1 = initial_fronts[0]

    history_objectives = [objectives.copy()]
    history_labels = [labels_pop.copy()]
    history_hv = [hypervolume_from_origin(objectives[:, 0], objectives[:, 1])] if len(objectives) else []
    history_hv_f1 = [hypervolume_from_origin(objectives[initial_f1, 0], objectives[initial_f1, 1])] if len(objectives) else []

    gen = 0
    while solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls and len(population) >= 2:
        gen += 1
        pop_size_actual = len(population)

        # ── Selección + offspring (idéntico a run_nsga2) ────────────────────
        ranks, crowding, _ = compute_ranks_and_crowding(objectives)
        parents = binary_tournament_selection(population, ranks, crowding, n_offspring=pop_size_actual, rng=rng)

        existing_sets = {frozenset(ind.tolist()) for ind in population}

        offspring_candidates = np.empty_like(parents)
        for i in range(0, pop_size_actual - 1, 2):
            c1, c2 = k_point_crossover(parents[i], parents[i + 1], n=n, crossover_prob=crossover_prob, rng=rng)
            c1 = controller_random_mutation(c1, n=n, mutation_prob=mutation_prob, rng=rng)
            c2 = controller_random_mutation(c2, n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[i] = _ensure_unique(c1, existing_sets, n, k, rng)
            offspring_candidates[i + 1] = _ensure_unique(c2, existing_sets, n, k, rng)
        if pop_size_actual % 2 == 1:
            c_last = controller_random_mutation(parents[-1].copy(), n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[-1] = _ensure_unique(c_last, existing_sets, n, k, rng)

        offspring, offspring_objectives, offspring_labels = evaluate_population(
            offspring_candidates, deb_matrix, dbb_matrix, max_obj_function_calls=max_obj_function_calls,
        )
        if len(offspring) == 0:
            break

        combined_population = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])
        combined_labels = np.vstack([labels_pop, offspring_labels])

        # ── PLS: ÚNICO agregado respecto a run_nsga2 ────────────────────────
        # Diversificación sobre el frente no dominado de Rv = Pv ∪ Qv.
        if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls:
            rv_fronts = non_dominated_sort(combined_objectives)
            f1_idx = rv_fronts[0]
            f1_pop = combined_population[f1_idx]

            if len(f1_pop) >= 1:
                pls_pop, pls_obj_list = pareto_local_search(
                    initial_population=f1_pop,
                    initial_objectives=combined_objectives[f1_idx],
                    deb_matrix=deb_matrix, dbb_matrix=dbb_matrix,
                    max_obj_calls=max_obj_function_calls,
                    local_budget=pls_local_budget,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    verbose=False,
                )

                if len(pls_pop) > 0:
                    # Solo agregar las soluciones REALMENTE nuevas que PLS
                    # descubrió (evita duplicar lo que ya estaba en Rv).
                    existing_keys = {frozenset(sol.tolist()) for sol in combined_population}
                    new_mask = np.array([frozenset(sol.tolist()) not in existing_keys for sol in pls_pop])
                    if new_mask.any():
                        new_solutions = pls_pop[new_mask]
                        new_objectives = np.asarray(pls_obj_list, dtype=np.float64)[new_mask]
                        # Los objetivos YA fueron evaluados dentro de PLS
                        # (no volver a llamar xie_beni/evaluate_population,
                        # eso duplicaría el gasto de cuota); solo se
                        # recalculan las asignaciones de cluster, que no
                        # consumen presupuesto.
                        new_labels = build_clusters_population(deb_matrix, new_solutions)

                        combined_population = np.vstack([combined_population, new_solutions])
                        combined_objectives = np.vstack([combined_objectives, new_objectives])
                        combined_labels = np.vstack([combined_labels, new_labels])

        # ── Selección de sobrevivientes (idéntica a run_nsga2) ──────────────
        _, combined_crowding, combined_fronts = compute_ranks_and_crowding(combined_objectives)

        target_size = min(pop_size, len(combined_population))
        next_indices: list[int] = []
        for front in combined_fronts:
            if len(next_indices) + len(front) <= target_size:
                next_indices.extend(front)
            else:
                remaining_slots = target_size - len(next_indices)
                front_sorted = sorted(front, key=lambda idx: combined_crowding[idx], reverse=True)
                next_indices.extend(front_sorted[:remaining_slots])
                break

        population = combined_population[next_indices]
        objectives = combined_objectives[next_indices]
        labels_pop = combined_labels[next_indices]

        current_fronts = non_dominated_sort(objectives)
        f1_indices = current_fronts[0]
        f1_hv = hypervolume_from_origin(objectives[f1_indices, 0], objectives[f1_indices, 1])

        history_objectives.append(objectives.copy())
        history_labels.append(labels_pop.copy())
        history_hv.append(hypervolume_from_origin(objectives[:, 0], objectives[:, 1]))
        history_hv_f1.append(f1_hv)

        mean_hv = np.mean(history_hv[-1])
        mean_hv_f1 = np.mean(f1_hv)
        print(
            f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  "
            f"población = {len(population)}  |  F1 = {len(f1_indices)}  |  "
            f"HV promedio = {mean_hv:.6f}  |  HV F1 promedio = {mean_hv_f1:.6f}"
        )

    print(f"\nCriterio de paro EXACTO: {solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_function_calls} llamadas a la función objetivo, {gen} generaciones completadas.")

    return {
        "population": population,
        "objectives": objectives,
        "labels": labels_pop,
        "fronts": non_dominated_sort(objectives),
        "history_objectives": history_objectives,
        "history_labels": history_labels,
        "history_hv": history_hv,
        "history_hv_mean": [float(np.mean(hv)) for hv in history_hv],
        "history_hv_f1": history_hv_f1,
        "history_hv_f1_mean": [float(np.mean(hv)) for hv in history_hv_f1],
        "generations_run": gen,
    }


#######################################################
# run_nsga2_MOLS: variante de run_nsga2 que agrega N-MOLS o L-MOLS
# (según `mode`) como estrategia de búsqueda local
#######################################################

def run_nsga2_MOLS(
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    mode: str,
    neighborhood: float,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    mols_local_budget: int | None = None,
    seed: int | None = None,
) -> dict:
    """
    Variante de run_nsga2 que agrega L-MOLS o N-MOLS (Haidine & Lehnert,
    2008; Toledo 2021) como estrategia de búsqueda local: tras construir
    Rv = Pv ∪ Qv, se extrae su frente no dominado F1 y se aplica el
    operador elegido (`mode`); las soluciones NUEVAS se agregan de vuelta
    a Rv antes de la selección de sobrevivientes. Resto del ciclo idéntico
    a run_nsga2 — MOPR/PLS no se agregan en esta variante.

    Parámetros
    ----------
    mode                : "l_mols" | "n_mols".
    neighborhood        : float — radio de vecindario (parámetro del
                          algoritmo, ver Sección 3.3.6 de Toledo).
    mols_local_budget   : int | None — cuota POR GENERACIÓN para MOLS (ver
                          run_nsga2_PLS — sin esto, MOLS puede consumir
                          TODO el presupuesto en una sola generación).

    Criterio de paro EXACTO: MOLS comparte la MISMA cuota global
    (solution_encoding.OBJ_FUNCTION_CALLS vs max_obj_function_calls).
    """
    assert mode in ("l_mols", "n_mols"), "mode debe ser 'l_mols' o 'n_mols'."
    mols_fn = l_mols if mode == "l_mols" else n_mols

    # M_V es estática (no depende de la población) -> se calcula UNA sola vez.
    m_v_matrix = build_neighborhood_matrix(deb_matrix, dbb_matrix)

    rng = np.random.default_rng(seed)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, deb_matrix, dbb_matrix, max_obj_function_calls=max_obj_function_calls,
    )

    initial_fronts = non_dominated_sort(objectives) if len(objectives) else [[]]
    initial_f1 = initial_fronts[0]

    history_objectives = [objectives.copy()]
    history_labels = [labels_pop.copy()]
    history_hv = [hypervolume_from_origin(objectives[:, 0], objectives[:, 1])] if len(objectives) else []
    history_hv_f1 = [hypervolume_from_origin(objectives[initial_f1, 0], objectives[initial_f1, 1])] if len(objectives) else []

    gen = 0
    while solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls and len(population) >= 2:
        gen += 1
        pop_size_actual = len(population)

        # ── Selección + offspring (idéntico a run_nsga2) ────────────────────
        ranks, crowding, _ = compute_ranks_and_crowding(objectives)
        parents = binary_tournament_selection(population, ranks, crowding, n_offspring=pop_size_actual, rng=rng)

        existing_sets = {frozenset(ind.tolist()) for ind in population}

        offspring_candidates = np.empty_like(parents)
        for i in range(0, pop_size_actual - 1, 2):
            c1, c2 = k_point_crossover(parents[i], parents[i + 1], n=n, crossover_prob=crossover_prob, rng=rng)
            c1 = controller_random_mutation(c1, n=n, mutation_prob=mutation_prob, rng=rng)
            c2 = controller_random_mutation(c2, n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[i] = _ensure_unique(c1, existing_sets, n, k, rng)
            offspring_candidates[i + 1] = _ensure_unique(c2, existing_sets, n, k, rng)
        if pop_size_actual % 2 == 1:
            c_last = controller_random_mutation(parents[-1].copy(), n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[-1] = _ensure_unique(c_last, existing_sets, n, k, rng)

        offspring, offspring_objectives, offspring_labels = evaluate_population(
            offspring_candidates, deb_matrix, dbb_matrix, max_obj_function_calls=max_obj_function_calls,
        )
        if len(offspring) == 0:
            break

        combined_population = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])
        combined_labels = np.vstack([labels_pop, offspring_labels])

        # ── MOLS: ÚNICO agregado respecto a run_nsga2 ───────────────────────
        if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls:
            rv_fronts = non_dominated_sort(combined_objectives)
            f1_idx = rv_fronts[0]
            f1_pop = combined_population[f1_idx]

            if len(f1_pop) >= 1:
                mols_pop, mols_obj = mols_fn(
                    f1_pop, deb_matrix, dbb_matrix, m_v_matrix, neighborhood,
                    max_obj_calls=max_obj_function_calls,
                    population_objectives=combined_objectives[f1_idx],
                    local_budget=mols_local_budget,
                    verbose=False,
                )

                if len(mols_pop) > 0:
                    existing_keys = {frozenset(sol.tolist()) for sol in combined_population}
                    new_mask = np.array([frozenset(sol.tolist()) not in existing_keys for sol in mols_pop])
                    if new_mask.any():
                        new_solutions = mols_pop[new_mask]
                        new_objectives = mols_obj[new_mask]
                        new_labels = build_clusters_population(deb_matrix, new_solutions)

                        combined_population = np.vstack([combined_population, new_solutions])
                        combined_objectives = np.vstack([combined_objectives, new_objectives])
                        combined_labels = np.vstack([combined_labels, new_labels])

        # ── Selección de sobrevivientes (idéntica a run_nsga2) ──────────────
        _, combined_crowding, combined_fronts = compute_ranks_and_crowding(combined_objectives)

        target_size = min(pop_size, len(combined_population))
        next_indices: list[int] = []
        for front in combined_fronts:
            if len(next_indices) + len(front) <= target_size:
                next_indices.extend(front)
            else:
                remaining_slots = target_size - len(next_indices)
                front_sorted = sorted(front, key=lambda idx: combined_crowding[idx], reverse=True)
                next_indices.extend(front_sorted[:remaining_slots])
                break

        population = combined_population[next_indices]
        objectives = combined_objectives[next_indices]
        labels_pop = combined_labels[next_indices]

        current_fronts = non_dominated_sort(objectives)
        f1_indices = current_fronts[0]
        f1_hv = hypervolume_from_origin(objectives[f1_indices, 0], objectives[f1_indices, 1])

        history_objectives.append(objectives.copy())
        history_labels.append(labels_pop.copy())
        history_hv.append(hypervolume_from_origin(objectives[:, 0], objectives[:, 1]))
        history_hv_f1.append(f1_hv)

        mean_hv = np.mean(history_hv[-1])
        mean_hv_f1 = np.mean(f1_hv)
        print(
            f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  "
            f"población = {len(population)}  |  F1 = {len(f1_indices)}  |  "
            f"HV promedio = {mean_hv:.6f}  |  HV F1 promedio = {mean_hv_f1:.6f}"
        )

    print(f"\nCriterio de paro EXACTO: {solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_function_calls} llamadas a la función objetivo, {gen} generaciones completadas.")

    return {
        "population": population,
        "objectives": objectives,
        "labels": labels_pop,
        "fronts": non_dominated_sort(objectives),
        "history_objectives": history_objectives,
        "history_labels": history_labels,
        "history_hv": history_hv,
        "history_hv_mean": [float(np.mean(hv)) for hv in history_hv],
        "history_hv_f1": history_hv_f1,
        "history_hv_f1_mean": [float(np.mean(hv)) for hv in history_hv_f1],
        "generations_run": gen,
    }
