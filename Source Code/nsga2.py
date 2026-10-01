"""
Ciclo principal de NSGA-II sobre la codificación de medoides. Ensambla
solution_encoding, operators, pareto_sorting y performance.
"""

############################
# Importaciones
############################
import numpy as np                  # Operaciones matemáticas eficientes.

# Importaciones propias (Asociado a Notes 1).
import solution_encoding
from solution_encoding  import random_medoids_pop, build_clusters_population, xie_beni_population
from pareto_sorting     import non_dominated_sort, crowding_distance
from operators          import binary_tournament_selection, k_point_crossover, controller_random_mutation
from performance        import hypervolume_from_origin

##################################
# Conjunto de funciones.
##################################

def evaluate_population(
    population: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    max_obj_function_calls: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa (XB_GE, XB_BI) para una población de medoides.

    Si max_obj_function_calls no es None, evalúa como máximo
    (max_obj_function_calls - OBJ_FUNCTION_CALLS) individuos, truncando
    la población de entrada si es necesario — garantiza que el total de
    llamadas a la función objetivo nunca supere el máximo, ni por una sola.

    Retorna
    -------
    evaluated_population : np.ndarray — subconjunto de `population` realmente evaluado (puede ser más chico que la entrada).
    objectives            : np.ndarray shape (m, 2).
    labels_pop            : np.ndarray shape (m, n).
    """
    # Tamaño de población.
    pop_size = population.shape[0]

    # Se comprueba cuota de llamadas a funciones objetivo.
    if max_obj_function_calls is not None:
        remaining = max_obj_function_calls - solution_encoding.OBJ_FUNCTION_CALLS
        pop_size = max(0, min(pop_size, remaining))

    # Población a ser evaluada.
    evaluated_population = population[:pop_size]

    # Evaluació final.
    if pop_size == 0:
        return evaluated_population, np.empty((0, 2)), np.empty((0, ge_matrix.shape[0]), dtype=np.int32)
    labels_pop = build_clusters_population(ge_matrix, evaluated_population)
    objectives = xie_beni_population(ge_matrix, bi_matrix, evaluated_population, labels_pop)

    return evaluated_population, objectives, labels_pop

def compute_ranks_and_crowding(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Convierte la salida de non_dominated_sort/crowding_distance (indexada
    por frente) en arreglos planos indexados por posición en la población,
    formato que requiere binary_tournament_selection.
    """
    fronts = non_dominated_sort(objectives)
    pop_size = len(objectives)
    ranks = np.empty(pop_size, dtype=np.int32)
    crowding = np.empty(pop_size, dtype=np.float64)

    for rank, front in enumerate(fronts):
        dist = crowding_distance(objectives, front)
        for idx in front:
            ranks[idx] = rank
            crowding[idx] = dist[idx]

    return ranks, crowding, fronts

def _ensure_unique(
    individual: np.ndarray,
    existing_sets: set[frozenset],
    n: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Si `individual` ya existe en `existing_sets` (mismo conjunto de medoides,
    sin importar el orden — un cromosoma es un conjunto, no una secuencia),
    lo reemplaza por un individuo nuevo generado al azar hasta encontrar uno
    que no esté repetido. Registra el resultado en `existing_sets` (in-place).

    Parámetros
    ----------
    individual    : np.ndarray — candidato a verificar, shape (k,).
    existing_sets : set[frozenset] — conjuntos de medoides ya presentes en
                    la población actual + candidatos ya aceptados en esta
                    misma generación (se modifica in-place).
    n, k          : int — tamaño del dataset y número de medoides.
    rng           : np.random.Generator.

    Retorna
    -------
    np.ndarray — el individuo original si era único, o un reemplazo aleatorio
    único en caso contrario.
    """
    key = frozenset(individual.tolist())

    while key in existing_sets:
        individual = rng.choice(n, size=k, replace=False).astype(individual.dtype)
        key = frozenset(individual.tolist())

    existing_sets.add(key)
    return individual

def run_nsga2(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Ciclo principal de NSGA-II. Criterio de paro EXACTO: nunca se ejecuta
    una llamada a la función objetivo por sobre max_obj_function_calls —
    evaluate_population trunca la última tanda (inicial u offspring) para
    respetar el presupuesto exactamente.
    """
    rng = np.random.default_rng(seed)

    # ── Población inicial P0 (puede truncarse si el presupuesto es muy bajo) ──
    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, ge_matrix, bi_matrix, max_obj_function_calls=max_obj_function_calls,
    )

    initial_fronts = non_dominated_sort(objectives) if len(objectives) else [[]]
    initial_f1 = initial_fronts[0]

    history_objectives = [objectives.copy()]
    history_labels     = [labels_pop.copy()]
    history_hv         = [hypervolume_from_origin(objectives[:, 0], objectives[:, 1])] if len(objectives) else []
    history_hv_f1      = [hypervolume_from_origin(objectives[initial_f1, 0], objectives[initial_f1, 1])] if len(objectives) else []

    gen = 0
    while solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls and len(population) >= 2:
        gen += 1
        pop_size_actual = len(population)

        # 1. Ranking + crowding de la población actual.
        ranks, crowding, _ = compute_ranks_and_crowding(objectives)

        # 2. Selección de padres (torneo binario).
        parents = binary_tournament_selection(population, ranks, crowding, n_offspring=pop_size_actual, rng=rng)

        # 3. Crossover + mutación → candidatos a descendencia Qv (aún sin evaluar).
        existing_sets = {frozenset(ind.tolist()) for ind in population}

        offspring_candidates = np.empty_like(parents)
        for i in range(0, pop_size_actual - 1, 2):
            c1, c2 = k_point_crossover(parents[i], parents[i + 1], n=n, crossover_prob=crossover_prob, rng=rng)
            c1 = controller_random_mutation(c1, n=n, mutation_prob=mutation_prob, rng=rng)
            c2 = controller_random_mutation(c2, n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[i]     = _ensure_unique(c1, existing_sets, n, k, rng)
            offspring_candidates[i + 1] = _ensure_unique(c2, existing_sets, n, k, rng)
        if pop_size_actual % 2 == 1:
            c_last = controller_random_mutation(parents[-1].copy(), n=n, mutation_prob=mutation_prob, rng=rng)
            offspring_candidates[-1] = _ensure_unique(c_last, existing_sets, n, k, rng)

        # 4. Evaluación truncada exactamente al presupuesto restante.
        offspring, offspring_objectives, offspring_labels = evaluate_population(
            offspring_candidates, ge_matrix, bi_matrix, max_obj_function_calls=max_obj_function_calls,
        )

        if len(offspring) == 0:
            break  # presupuesto agotado antes de evaluar algún hijo

        # 5. Rv = Pv ∪ Qv (evaluada).
        combined_population = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])
        combined_labels     = np.vstack([labels_pop, offspring_labels])

        # 6. Ranking + crowding de Rv, y construcción de Pv+1.
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
        print(f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  población = {len(population)}  |  F1 = {len(f1_indices)}  |  HV promedio = {mean_hv:.6f}  |  HV F1 promedio = {mean_hv_f1:.6f}")

    print(f"\nCriterio de paro EXACTO: {solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_function_calls} llamadas a la función objetivo, {gen} generaciones completadas.")

    return {
        "population":         population,
        "objectives":         objectives,
        "labels":             labels_pop,
        "fronts":             non_dominated_sort(objectives),
        "history_objectives": history_objectives,
        "history_labels":     history_labels,
        "history_hv":         history_hv,
        "history_hv_mean":    [float(np.mean(hv)) for hv in history_hv],
        "history_hv_f1":      history_hv_f1,
        "history_hv_f1_mean": [float(np.mean(hv)) for hv in history_hv_f1],
        "generations_run":    gen,
    }