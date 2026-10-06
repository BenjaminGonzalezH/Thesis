"""
Variantes de run_nsga2 desarrolladas en Notes_3-BIOperators.ipynb, que
agregan cada uno de los operadores de búsqueda local basados en información
biológica como estrategia aplicada sobre el frente no dominado F1 de
Rv = Pv ∪ Qv, antes de la selección de sobrevivientes.

- run_nsga2_BCS : agrega Bio-Counterweight Search.
- run_nsga2_MBS : agrega Medoid-Based Search (regímenes bio / no_discrepancy / expresion).

El resto del ciclo (selección por torneo, crossover, mutación, truncamiento
por rank+crowding) es IDÉNTICO a run_nsga2 (nsga2.py). Todas comparten la
MISMA cuota global (solution_encoding.OBJ_FUNCTION_CALLS vs
max_obj_function_calls) con los operadores, por lo que nunca se excede
max_obj_function_calls.

Este módulo también contiene los operadores BCS y MBS (con tope de
cuota opcional `max_obj_calls`), ya que las variantes dependen de ellos.
"""

import numpy as np

import solution_encoding
from solution_encoding import random_medoids_pop, build_clusters_population, xie_beni, xie_beni_population
from pareto_sorting import non_dominated_sort
from operators import binary_tournament_selection, k_point_crossover, controller_random_mutation
from performance import hypervolume_from_origin
from nsga2 import evaluate_population, compute_ranks_and_crowding, _ensure_unique

#######################################################
# MODULO BIO-COUNTERWEIGHT SEARCH (BCS)
# Para cada medoide z_k de la solución se busca un "hub biológico" z_bio
# dentro de su propio cluster (el gen con menor distancia BI PROMEDIO al
# resto del cluster) y se explora el vecindario por percentil combinando
# ambas anclas (z_k y z_bio). Solo se aceptan reemplazos que MEJOREN
# estrictamente ambos índices Xie-Beni respecto a la solución original.
#######################################################

def _biological_hub(cluster_members: np.ndarray, zk: int, bi_matrix: np.ndarray) -> int:
    """
    Determina z_bio: el gen de C_k \\ {z_k} con menor distancia BI PROMEDIO
    al resto del cluster (bloque "Hub biológico" del pseudocódigo).

    Nota de rendimiento
    --------------------
    Se vectoriza el promedio completo con una sola indexación de submatriz
    en vez de un loop de Python con un `.mean()` por candidato (uno por
    gen del cluster, en cada llamada por medoide). `candidates` es siempre
    un subconjunto de `cluster_members` (se excluye solo z_k), por lo que
    cada fila de `sub` tiene EXACTAMENTE una autocomparación (g == g);
    se resta esa entrada de la suma y se divide por |C_k| - 1 en bloque.
    """
    candidates = cluster_members[cluster_members != zk]
    sub = bi_matrix[np.ix_(candidates, cluster_members)]          # (n_cand, n_cluster)
    self_mask = candidates[:, None] == cluster_members[None, :]    # una True por fila
    prom_bio = (sub.sum(axis=1) - sub[self_mask]) / (len(cluster_members) - 1)
    return int(candidates[np.argmin(prom_bio)])

def _percentile_neighborhood(
    anchor: int,
    d_avg: np.ndarray,
    excluir: set[int],
    p: float,
) -> set[int]:
    """
    N(ancla): índices fuera de `excluir` cuya distancia D_avg al ancla es
    menor o igual al percentil `p` de esas mismas distancias.

    Nota de rendimiento
    --------------------
    `elegibles` se obtiene con una máscara booleana (vectorizada en C) en
    vez de un list comprehension de Python sobre range(n): esta función se
    llama hasta 2 veces por medoide (hasta 2K veces por llamada a BCS) y n
    es del orden de miles de genes, por lo que el loop puro de Python era
    el costo dominante frente al resto de la función, ya vectorizada.
    """
    mask = np.ones(d_avg.shape[0], dtype=bool)
    mask[list(excluir)] = False
    elegibles = np.where(mask)[0]
    dists = d_avg[anchor, elegibles]
    umbral = np.percentile(dists, p * 100)
    return set(elegibles[dists <= umbral].tolist())

def bio_counterweight_search(
    C: np.ndarray,
    labels: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    p: float = 0.05,
    verbose: bool = False,
    max_obj_calls: int | None = None,
    base_objectives: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Bio-Counterweight search (BCS): para cada medoide z_k de C se determina
    un hub biológico z_bio dentro de su propio cluster y se explora el
    vecindario por percentil `p`, combinando ambas anclas (z_k, z_bio). Solo
    se aceptan reemplazos que mejoren estrictamente AMBOS índices Xie-Beni
    (XB_EB y XB_BB) respecto a la solución original.

    Parámetros
    ----------
    C          : np.ndarray (k,) — medoides de la solución actual.
    labels     : np.ndarray (n,) — partición ÚNICA ya evaluada (1-based,
                 mismo formato que build_clusters_population).
    ge_matrix, bi_matrix : matrices de distancia n×n, YA normalizadas en [0,1].
    p          : percentil de vecindad (ej. 0.05 → 5% más cercanos).

    max_obj_calls   : int | None — tope sobre solution_encoding.OBJ_FUNCTION_CALLS
        (misma cuota GLOBAL de nsga2.py). Si es None no hay tope. Si el
        presupuesto no alcanza para un batch completo, éste se recorta
        (al azar, con `rng`) y el operador se detiene al agotarse.
    base_objectives : (XB_EB_0, XB_BB_0) | None — objetivos ya conocidos de C;
        evita re-evaluarlos (1 llamada) cuando el NSGA-II ya los tiene.
    rng             : np.random.Generator | None — para recortar candidatas.

    Retorna
    -------
    list[dict] con llaves "solution", "labels", "xb_eb", "xb_bb" — una
    entrada por cada candidata ACEPTADA (puede quedar vacía).

    Nota de rendimiento
    --------------------
    Las candidatas C_prime de cada medoide z_k se apilan en un solo batch y
    se evalúan con UNA llamada a build_clusters_population + UNA llamada a
    xie_beni_population, en vez de reconstruir la partición y recalcular
    Xie-Beni candidata por candidata (evita 2·|N(C)| llamadas redundantes
    a numpy por cada medoide, sin cambiar qué candidatas se evalúan).
    """
    K = len(C)
    d_avg = (ge_matrix + bi_matrix) / 2                      # Sin normalización adicional.
    excluir = set(C.tolist())                                # TODOS los medoides actuales de la solución.

    if base_objectives is None:
        xb_eb_0, xb_bb_0 = xie_beni(ge_matrix, bi_matrix, C, labels)
    else:
        xb_eb_0, xb_bb_0 = base_objectives
    if rng is None:
        rng = np.random.default_rng()
    aceptadas: list[dict] = []

    for k in range(1, K + 1):
        zk = int(C[k - 1])
        cluster_members = np.where(labels == k)[0]

        if len(cluster_members) <= 1:
            continue                                          # Cluster trivial → se evita.

        # ── Hub biológico: minimiza la distancia BI promedio dentro de C_k ──
        z_bio = _biological_hub(cluster_members, zk, bi_matrix)

        # ── Vecindarios por percentil sobre TODO el dataset, excluyendo C ──
        N_zk = _percentile_neighborhood(zk, d_avg, excluir, p)

        if z_bio == zk:
            N_C = N_zk                                        # Misma ancla → mismo cálculo, sin caso especial.
        else:
            N_zbio = _percentile_neighborhood(z_bio, d_avg, excluir, p)
            N_C = N_zk & N_zbio
            if not N_C:
                N_C = N_zbio                                  # Respaldo: prioriza el ancla BIOLÓGICA.

        if verbose:
            print(f"[BCS] k={k}  z_k={zk}  z_bio={z_bio}  |N(C)|={len(N_C)}")

        if not N_C:
            continue

        # ── Evaluación batched y aceptación estricta ─────────────────────
        candidatos = np.array(sorted(N_C), dtype=C.dtype)

        # ── Cuota global: recorte del batch y detención al agotarse ───────
        if max_obj_calls is not None:
            remaining = max_obj_calls - solution_encoding.OBJ_FUNCTION_CALLS
            if remaining <= 0:
                break
            if len(candidatos) > remaining:
                candidatos = rng.permutation(candidatos)[:remaining]
        C_primes = np.tile(C, (len(candidatos), 1))
        C_primes[:, k - 1] = candidatos

        labels_primes = build_clusters_population(ge_matrix, C_primes)
        xb_primes = xie_beni_population(ge_matrix, bi_matrix, C_primes, labels_primes)

        accept_mask = (xb_primes[:, 0] < xb_eb_0) & (xb_primes[:, 1] < xb_bb_0)
        for C_prime, labels_prime, (xb_eb_p, xb_bb_p), aceptado in zip(
            C_primes, labels_primes, xb_primes, accept_mask
        ):
            if aceptado:
                aceptadas.append({
                    "solution": C_prime.copy(),
                    "labels": labels_prime,
                    "xb_eb": float(xb_eb_p),
                    "xb_bb": float(xb_bb_p),
                })

    return aceptadas

#######################################################
# MODULO MEDOID-BASED SEARCH (MBS)
# Se construye una matriz de discrepancia Δ = S_EB - S_BB (con S = 1 - D)
# y, según el régimen `mode`, se transforma en una matriz de distancia
# D_mode sobre la cual se reutiliza build_clusters_population para definir
# el vecindario de reemplazo de cada medoide: los genes cuyo medoide más
# cercano (bajo D_mode) es z_k, excluyendo los medoides de la solución.
# La aceptación es idéntica a BCS (mejora estricta de ambos Xie-Beni).
#######################################################

MBS_MODES = ("bio", "no_discrepancy", "expresion")

def build_discrepancy_matrix(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Construye D_mode a partir de Δ(i,j) = S_EB(i,j) - S_BB(i,j) ∈ [-1, 1],
    con S = 1 - D:

        bio             → Δ + 1     (cercanía = alta relación biológica pese a baja expresión)
        expresion       → 1 - Δ     (régimen inverso)
        no_discrepancy  → |Δ|       (cercanía = concordancia entre ambos criterios)
    """
    assert mode in MBS_MODES, f"mode debe ser uno de {MBS_MODES}."
    delta = (1.0 - ge_matrix) - (1.0 - bi_matrix)
    if mode == "bio":
        return delta + 1.0
    if mode == "expresion":
        return 1.0 - delta
    return np.abs(delta)

def medoid_based_search(
    C: np.ndarray,
    labels: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    mode: str = "bio",
    d_mode: np.ndarray | None = None,
    verbose: bool = False,
    max_obj_calls: int | None = None,
    base_objectives: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Medoid-Based Search (MBS): para cada medoide z_k de C, su vecindario
    N(z_k) son los genes cuyo medoide más cercano bajo D_mode es z_k
    (excluyendo los medoides de C). Cada candidato c ∈ N(z_k) reemplaza a
    z_k y se acepta solo si mejora estrictamente AMBOS índices Xie-Beni
    (partición reconstruida globalmente con D_EB).

    Parámetros
    ----------
    C          : np.ndarray (k,) — medoides de la solución actual.
    labels     : np.ndarray (n,) — partición ÚNICA ya evaluada (1-based).
    ge_matrix, bi_matrix : matrices de distancia n×n, YA normalizadas en [0,1].
    mode       : "bio" | "no_discrepancy" | "expresion" (fijo por corrida).
    d_mode     : np.ndarray | None — D_mode precalculada; si es None se
        construye aquí. Conviene precalcularla al aplicar el operador
        sobre varias soluciones, ya que no depende de C.

    max_obj_calls   : int | None — tope sobre solution_encoding.OBJ_FUNCTION_CALLS
        (misma cuota GLOBAL de nsga2.py). Si es None no hay tope. Si el
        presupuesto no alcanza para un batch completo, éste se recorta
        (al azar, con `rng`) y el operador se detiene al agotarse.
    base_objectives : (XB_EB_0, XB_BB_0) | None — objetivos ya conocidos de C;
        evita re-evaluarlos (1 llamada) cuando el NSGA-II ya los tiene.
    rng             : np.random.Generator | None — para recortar candidatas.

    Retorna
    -------
    list[dict] con llaves "solution", "labels", "xb_eb", "xb_bb" — una
    entrada por cada candidata ACEPTADA (puede quedar vacía).

    Nota de rendimiento
    --------------------
    Igual que en BCS: las candidatas C_prime de cada medoide z_k se apilan
    en un solo batch y se evalúan con UNA llamada a build_clusters_population
    + UNA llamada a xie_beni_population, en vez de candidata por candidata.
    Además, N(z_k) se filtra con np.isin (vectorizado en C) en vez de un
    list comprehension de Python sobre los índices del cluster, ya que este
    filtrado se repite una vez por medoide (K veces por llamada a MBS).
    """
    if d_mode is None:
        d_mode = build_discrepancy_matrix(ge_matrix, bi_matrix, mode)

    K = len(C)
    excluir = set(C.tolist())

    if base_objectives is None:
        xb_eb_0, xb_bb_0 = xie_beni(ge_matrix, bi_matrix, C, labels)
    else:
        xb_eb_0, xb_bb_0 = base_objectives
    if rng is None:
        rng = np.random.default_rng()
    aceptadas: list[dict] = []

    # Asignación de cada gen a su medoide más cercano bajo D_mode.
    labels_mode = build_clusters_population(d_mode, C[None, :])[0]

    for k in range(1, K + 1):
        zk = int(C[k - 1])
        candidatos_zk = np.where(labels_mode == k)[0]
        N_zk = candidatos_zk[~np.isin(candidatos_zk, list(excluir))].tolist()

        if verbose:
            print(f"[MBS-{mode}] k={k}  z_k={zk}  |N(z_k)|={len(N_zk)}")

        if not N_zk:
            continue

        # ── Evaluación batched y aceptación estricta ─────────────────────
        candidatos = np.array(N_zk, dtype=C.dtype)

        # ── Cuota global: recorte del batch y detención al agotarse ───────
        if max_obj_calls is not None:
            remaining = max_obj_calls - solution_encoding.OBJ_FUNCTION_CALLS
            if remaining <= 0:
                break
            if len(candidatos) > remaining:
                candidatos = rng.permutation(candidatos)[:remaining]
        C_primes = np.tile(C, (len(candidatos), 1))
        C_primes[:, k - 1] = candidatos

        labels_primes = build_clusters_population(ge_matrix, C_primes)
        xb_primes = xie_beni_population(ge_matrix, bi_matrix, C_primes, labels_primes)

        accept_mask = (xb_primes[:, 0] < xb_eb_0) & (xb_primes[:, 1] < xb_bb_0)
        for C_prime, labels_prime, (xb_eb_p, xb_bb_p), aceptado in zip(
            C_primes, labels_primes, xb_primes, accept_mask
        ):
            if aceptado:
                aceptadas.append({
                    "solution": C_prime.copy(),
                    "labels": labels_prime,
                    "xb_eb": float(xb_eb_p),
                    "xb_bb": float(xb_bb_p),
                })

    return aceptadas

#######################################################
# Núcleo común de run_nsga2_BCS / run_nsga2_MBS: ciclo de
# run_nsga2 + un operador BI aplicado sobre el F1 de Rv = Pv ∪ Qv
#######################################################

def _run_nsga2_BI(
    apply_operator,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    crossover_prob: float,
    mutation_prob: float,
    local_budget: int | None,
    seed: int | None,
) -> dict:
    """
    Ciclo de run_nsga2 con UN único agregado: tras construir Rv = Pv ∪ Qv se
    extrae su frente no dominado F1 y se aplica `apply_operator` a cada una
    de sus soluciones (en orden aleatorio, para que una cuota escasa no
    favorezca siempre a las mismas); las soluciones NUEVAS que aceptan se
    agregan de vuelta a Rv antes de la selección de sobrevivientes. Lo demás
    (torneo, crossover, mutación, truncamiento por rank+crowding) es
    IDÉNTICO a run_nsga2.

    apply_operator(C, labels, objectives, max_obj_calls, rng, population)
        → list[dict] con llaves "solution", "labels", "xb_eb", "xb_bb".
        `population` es Rv (por si el operador mide algo sobre la población).

    local_budget : int | None — cuota de evaluaciones que el operador puede
        consumir POR GENERACIÓN (mismo criterio que pls_local_budget /
        mols_local_budget en Notes_2). Sin ella el operador puede agotar
        toda la cuota global en la primera generación.

    Criterio de paro EXACTO: el operador comparte la MISMA cuota global
    (solution_encoding.OBJ_FUNCTION_CALLS vs max_obj_function_calls).
    """
    rng = np.random.default_rng(seed)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, ge_matrix, bi_matrix, max_obj_function_calls=max_obj_function_calls,
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
            offspring_candidates, ge_matrix, bi_matrix, max_obj_function_calls=max_obj_function_calls,
        )
        if len(offspring) == 0:
            break

        combined_population = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])
        combined_labels = np.vstack([labels_pop, offspring_labels])

        # ── Operador BI: ÚNICO agregado respecto a run_nsga2 ────────────────
        if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls:
            # Tope de ESTA generación = mínimo entre cuota global y cuota local.
            gen_limit = max_obj_function_calls if local_budget is None else min(
                max_obj_function_calls, solution_encoding.OBJ_FUNCTION_CALLS + local_budget
            )
            f1_idx = non_dominated_sort(combined_objectives)[0]
            existing_keys = {frozenset(sol.tolist()) for sol in combined_population}

            new_solutions, new_objectives, new_labels = [], [], []
            for j in rng.permutation(f1_idx):
                if solution_encoding.OBJ_FUNCTION_CALLS >= gen_limit:
                    break
                aceptadas = apply_operator(
                    combined_population[j], combined_labels[j], combined_objectives[j],
                    gen_limit, rng, combined_population,
                )
                for d in aceptadas:
                    # Solo soluciones REALMENTE nuevas (ni en Rv ni ya agregadas).
                    key = frozenset(d["solution"].tolist())
                    if key in existing_keys:
                        continue
                    existing_keys.add(key)
                    new_solutions.append(d["solution"])
                    new_objectives.append((d["xb_eb"], d["xb_bb"]))
                    new_labels.append(d["labels"])

            if new_solutions:
                # Objetivos y labels YA calculados dentro del operador (no
                # re-evaluar: duplicaría el gasto de cuota).
                combined_population = np.vstack([combined_population, np.asarray(new_solutions, dtype=combined_population.dtype)])
                combined_objectives = np.vstack([combined_objectives, np.asarray(new_objectives, dtype=np.float64)])
                combined_labels = np.vstack([combined_labels, np.asarray(new_labels, dtype=combined_labels.dtype)])

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
# run_nsga2_BCS: variante de run_nsga2 que agrega Bio-Counterweight
# Search (BCS) como estrategia de búsqueda local
#######################################################

def run_nsga2_BCS(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    p: float = 0.05,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    bcs_local_budget: int | None = None,
    seed: int | None = None,
) -> dict:
    """
    Variante de run_nsga2 que agrega BCS: tras construir Rv = Pv ∪ Qv, se
    aplica BCS (percentil de vecindad `p`) sobre cada solución de su F1; las
    reemplazos aceptados (mejoran estrictamente ambos Xie-Beni) se agregan
    a Rv antes de la selección de sobrevivientes. Ver _run_nsga2_BI.
    """
    def apply_operator(C, labels, objs, limit, rng, _population):
        return bio_counterweight_search(
            C, labels, ge_matrix, bi_matrix, p=p,
            max_obj_calls=limit, base_objectives=tuple(objs), rng=rng,
        )

    return _run_nsga2_BI(
        apply_operator, ge_matrix, bi_matrix, n, k, pop_size, max_obj_function_calls,
        crossover_prob, mutation_prob, bcs_local_budget, seed,
    )

#######################################################
# run_nsga2_MBS: variante de run_nsga2 que agrega Medoid-Based Search
# (MBS, régimen `mode`) como estrategia de búsqueda local
#######################################################

def run_nsga2_MBS(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    mode: str = "bio",
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    mbs_local_budget: int | None = None,
    seed: int | None = None,
) -> dict:
    """
    Variante de run_nsga2 que agrega MBS: tras construir Rv = Pv ∪ Qv, se
    aplica MBS sobre cada solución de su F1; los reemplazos aceptados
    (mejoran estrictamente ambos Xie-Beni) se agregan a Rv antes de la
    selección de sobrevivientes. Ver _run_nsga2_BI.

    mode : "bio" | "no_discrepancy" | "expresion" — fijo por corrida. D_mode
           no depende de la población, así que se construye UNA sola vez.
    """
    assert mode in MBS_MODES, f"mode debe ser uno de {MBS_MODES}."
    d_mode = build_discrepancy_matrix(ge_matrix, bi_matrix, mode)

    def apply_operator(C, labels, objs, limit, rng, _population):
        return medoid_based_search(
            C, labels, ge_matrix, bi_matrix, mode=mode, d_mode=d_mode,
            max_obj_calls=limit, base_objectives=tuple(objs), rng=rng,
        )

    return _run_nsga2_BI(
        apply_operator, ge_matrix, bi_matrix, n, k, pop_size, max_obj_function_calls,
        crossover_prob, mutation_prob, mbs_local_budget, seed,
    )
