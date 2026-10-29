"""
Variantes de run_nsga2 desarrolladas en Notes_3-BIOperators.ipynb, que
agregan cada uno de los operadores de búsqueda local basados en información
biológica como estrategia aplicada sobre el frente no dominado F1 de
Rv = Pv ∪ Qv, antes de la selección de sobrevivientes.

- run_nsga2_PNS : agrega Pareto-Neighborhood Search.
- run_nsga2_DBS : agrega Discrepancy-Based Search (regímenes bio / no_discrepancy / expresion).

El resto del ciclo (selección por torneo, crossover, mutación, truncamiento
por rank+crowding) es IDÉNTICO a run_nsga2 (nsga2.py). Todas comparten la
MISMA cuota global (solution_encoding.OBJ_FUNCTION_CALLS vs
max_obj_function_calls) con los operadores, por lo que nunca se excede
max_obj_function_calls.

Este módulo también contiene los operadores PNS y DBS (con tope de
cuota opcional `max_obj_calls`), ya que las variantes dependen de ellos.
"""

import numpy as np

import solution_encoding
from solution_encoding import random_medoids_pop, build_clusters_population, xie_beni, xie_beni_population
from pareto_sorting import non_dominated_sort
from operators import binary_tournament_selection, k_point_crossover, controller_random_mutation
from performance import hypervolume_f1
from nsga2 import evaluate_population, compute_ranks_and_crowding, _ensure_unique

#######################################################
# MODULO PARETO-NEIGHBORHOOD SEARCH (PNS)
# Para cada medoide z_k de la solución se explora su vecindario Pareto:
# los primeros nF frentes de no dominancia, sobre todo el dataset (sin los
# medoides de C), respecto a la cercanía (D_EB, D_BB) a z_k. Cada candidata
# reemplaza a z_k, la partición se reconstruye GLOBALMENTE y se acepta si
# MEJORA al menos uno de los índices Xie-Beni (XB_EB o XB_BB). El conjunto
# aceptado se ordena por frentes de Pareto y se retorna solo su F1.
#######################################################

def _first_nF_fronts_2d(f_a: np.ndarray, f_b: np.ndarray, nF: int) -> np.ndarray:
    """
    Máscara booleana de los puntos que pertenecen a los primeros `nF` frentes
    de Pareto (minimización de ambas coordenadas) del conjunto (f_a, f_b).

    Equivale a `non_dominated_sort` (pareto_sorting.py) truncado a nF frentes,
    pero en O(nF · m) tras un único ordenamiento lexicográfico en vez de
    O(m²) en tiempo y memoria: aquí m ≈ n (miles de genes) y la función se
    llama hasta 2K veces por solución, por lo que la matriz de dominancia
    m×m×2 de non_dominated_sort resulta prohibitiva.

    Los puntos idénticos no se dominan entre sí (mismo frente), igual que en
    `_dominates`. Sobre los puntos únicos ordenados lexicográficamente, i es
    dominado ⟺ existe un punto anterior j con f_b[j] <= f_b[i].
    """
    m = len(f_a)
    order = np.lexsort((f_b, f_a))                                 # clave primaria f_a, secundaria f_b.
    a_s, b_s = f_a[order], f_b[order]

    es_nuevo = np.ones(m, dtype=bool)
    es_nuevo[1:] = (a_s[1:] != a_s[:-1]) | (b_s[1:] != b_s[:-1])
    grupo = np.cumsum(es_nuevo) - 1                                # punto único de cada elemento.
    ub = b_s[es_nuevo]

    frente = np.full(len(ub), -1, dtype=np.int64)
    restantes = np.arange(len(ub))
    for r in range(nF):
        if len(restantes) == 0:
            break
        b_rest = ub[restantes]
        previo = np.empty_like(b_rest)
        previo[0] = np.inf
        previo[1:] = np.minimum.accumulate(b_rest)[:-1]
        no_dom = b_rest < previo
        frente[restantes[no_dom]] = r
        restantes = restantes[~no_dom]

    mask = np.zeros(m, dtype=bool)
    mask[order] = frente[grupo] >= 0
    return mask

def _pareto_neighborhood(
    anchor: int,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    excluir_mask: np.ndarray,
    nF: int,
) -> set[int]:
    """
    FrentesDeVecindad(a, D_GE, D_BI, excluir, nF): vecindario Pareto del
    ancla `a`. Cada gen elegible i (fuera de `excluir` e i ≠ a) se describe
    por (D_GE[a, i], D_BI[a, i]), ambos a minimizar, y se retiene la unión
    de los primeros `nF` frentes de Pareto F_1 ∪ ... ∪ F_nF.

    excluir_mask : np.ndarray (n,) bool — True en los medoides de C.
    """
    mask = ~excluir_mask
    mask[anchor] = False
    elegibles = np.where(mask)[0]
    sel = _first_nF_fronts_2d(ge_matrix[anchor, elegibles], bi_matrix[anchor, elegibles], nF)
    return set(elegibles[sel].tolist())

def pareto_neighborhood_search(
    C: np.ndarray,
    labels: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    nF: int = 1,
    verbose: bool = False,
    max_obj_calls: int | None = None,
    base_objectives: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Pareto-Neighborhood Search (PNS): se elige UN medoide z_k de C al azar
    (análogo a PLS) y se explora su vecindario Pareto (primeros `nF` frentes
    respecto a (D_EB, D_BB) a z_k, excluyendo todos los medoides de C). Cada
    candidata c reemplaza a z_k y se reconstruye la partición completa; se
    acepta si mejora AL MENOS UNO de los dos índices Xie-Beni respecto a la
    solución original. Se retornan todas las candidatas aceptadas.

    Parámetros
    ----------
    C          : np.ndarray (k,) — medoides de la solución actual.
    labels     : np.ndarray (n,) — partición ÚNICA ya evaluada (1-based,
                 mismo formato que build_clusters_population).
    ge_matrix, bi_matrix : matrices de distancia n×n, YA normalizadas en [0,1].
    nF         : número de frentes de Pareto que se conservan por medoide
                 (nF=1 → solo genes no dominados respecto a su cercanía
                 (D_EB, D_BB) a z_k).

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
    entrada por cada candidata del F1 del conjunto ACEPTADO (puede quedar vacía).

    Nota de rendimiento
    --------------------
    Las candidatas C_prime de cada medoide z_k se apilan en un solo batch y
    se evalúan con UNA llamada a build_clusters_population + UNA llamada a
    xie_beni_population, en vez de reconstruir la partición y recalcular
    Xie-Beni candidata por candidata.
    """
    K = len(C)
    excluir_mask = np.zeros(ge_matrix.shape[0], dtype=bool)
    excluir_mask[C] = True                                   # TODOS los medoides actuales de la solución.

    if base_objectives is None:
        xb_eb_0, xb_bb_0 = xie_beni(ge_matrix, bi_matrix, C, labels)
    else:
        xb_eb_0, xb_bb_0 = base_objectives
    if rng is None:
        rng = np.random.default_rng()

    # ── Un único medoide z_k elegido al azar (como PLS) ──
    k = int(rng.integers(1, K + 1))
    zk = int(C[k - 1])

    # ── Vecindario Pareto de z_k sobre TODO el dataset, excluyendo C ──
    N_zk = _pareto_neighborhood(zk, ge_matrix, bi_matrix, excluir_mask, nF)

    if verbose:
        print(f"[PNS] k={k}  z_k={zk}  |N(z_k)|={len(N_zk)}")

    if not N_zk:
        return []

    # ── Evaluación batched y aceptación (mejora de al menos un objetivo) ──
    candidatos = np.array(sorted(N_zk), dtype=C.dtype)

    # ── Cuota global: recorte del batch ───────
    if max_obj_calls is not None:
        remaining = max_obj_calls - solution_encoding.OBJ_FUNCTION_CALLS
        if remaining <= 0:
            return []
        if len(candidatos) > remaining:
            candidatos = rng.permutation(candidatos)[:remaining]
    C_primes = np.tile(C, (len(candidatos), 1))
    C_primes[:, k - 1] = candidatos

    labels_primes = build_clusters_population(ge_matrix, C_primes)
    xb_primes = xie_beni_population(ge_matrix, bi_matrix, C_primes, labels_primes)

    accept_mask = (xb_primes[:, 0] < xb_eb_0) | (xb_primes[:, 1] < xb_bb_0)
    return [
        {
            "solution": C_primes[i].copy(),
            "labels": labels_primes[i],
            "xb_eb": float(xb_primes[i, 0]),
            "xb_bb": float(xb_primes[i, 1]),
        }
        for i in np.flatnonzero(accept_mask)
    ]

#######################################################
# MODULO MEDOID-BASED SEARCH (DBS)
# Se construye una matriz de discrepancia logarítmica
#     L = log((D_BB + ε) / (D_EB + ε))
# y, según el régimen `mode`, se transforma en una matriz D_mode sobre la
# cual se reutiliza build_clusters_population para particionar los genes
# en "celdas" (medoide más cercano bajo D_mode). El vecindario de cada
# medoide z_k es el percentil `p` más cercano a z_k DENTRO de su celda
# (sin medoides de la solución). La aceptación es estricta (mejora
# de ambos Xie-Beni, partición reconstruida globalmente con D_EB); se
# retornan todas las candidatas aceptadas.
#######################################################

DBS_MODES = ("bio", "no_discrepancy", "expresion")
DBS_EPS = 1e-3          # ε por defecto de la matriz de discrepancia.

def build_discrepancy_matrix(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    mode: str,
    eps: float = DBS_EPS,
) -> np.ndarray:
    """
    Construye D_mode a partir de L = log((D_BB + ε) / (D_EB + ε)):

        L < 0 : más cerca en biología que en expresión
        L > 0 : más cerca en expresión que en biología
        L ≈ 0 : concordancia entre ambos criterios

        bio             → L     (cercanía = alta relación biológica pese a baja expresión)
        expresion       → -L    (régimen inverso)
        no_discrepancy  → |L|   (cercanía = concordancia entre ambos criterios)

    D_mode puede tomar valores negativos (es solo un criterio de orden).
    """
    assert mode in DBS_MODES, f"mode debe ser uno de {DBS_MODES}."
    L = np.log((bi_matrix + eps) / (ge_matrix + eps))
    if mode == "bio":
        return L
    if mode == "expresion":
        return -L
    return np.abs(L)

def discrepancy_based_search(
    C: np.ndarray,
    labels: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    mode: str = "bio",
    eps: float = DBS_EPS,
    p: float = 0.10,
    d_mode: np.ndarray | None = None,
    verbose: bool = False,
    max_obj_calls: int | None = None,
    base_objectives: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Discrepancy-Based Search (DBS): cada gen se asigna al medoide más cercano
    bajo D_mode (celdas). Para cada medoide z_k de C, Celda_k = genes de su
    celda sin los medoides de C, y su vecindario N(z_k) es el percentil `p`
    de Celda_k más cercano a z_k según D_mode. Cada candidato c ∈ N(z_k)
    reemplaza a z_k y se acepta solo si mejora estrictamente AMBOS índices
    Xie-Beni (partición reconstruida globalmente con D_EB). El operador
    retorna todas las candidatas aceptadas (de los K medoides).

    Parámetros
    ----------
    C          : np.ndarray (k,) — medoides de la solución actual.
    labels     : np.ndarray (n,) — partición ÚNICA ya evaluada (1-based).
    ge_matrix, bi_matrix : matrices de distancia n×n, YA normalizadas en [0,1].
    mode       : "bio" | "no_discrepancy" | "expresion" (fijo por corrida).
    eps        : ε de la matriz de discrepancia (evita división por 0 y log 0).
    p          : percentil de vecindad dentro de la celda de cada medoide
        (ej. 0.10 → 10% más cercano a z_k).
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
    entrada por cada candidata aceptada (puede quedar vacía).

    Nota de rendimiento
    --------------------
    Igual que en PNS: las candidatas C_prime de cada medoide z_k se apilan
    en un solo batch y se evalúan con UNA llamada a build_clusters_population
    + UNA llamada a xie_beni_population, en vez de candidata por candidata.
    Además, Celda_k se filtra con una máscara booleana (vectorizada en C)
    en vez de un list comprehension de Python sobre los índices de la celda.
    """
    if d_mode is None:
        d_mode = build_discrepancy_matrix(ge_matrix, bi_matrix, mode, eps)

    K = len(C)
    es_medoide = np.zeros(ge_matrix.shape[0], dtype=bool)
    es_medoide[C] = True                                     # TODOS los medoides actuales de la solución.

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
        celda_k = np.where((labels_mode == k) & ~es_medoide)[0]   # Celda completa, sin medoides.

        if len(celda_k) == 0:
            if verbose:
                print(f"[DBS-{mode}] k={k}  z_k={zk}  celda vacía")
            continue

        # ── Vecindario: percentil p más cercano a z_k DENTRO de la celda ──
        dists = d_mode[zk, celda_k]
        umbral = np.percentile(dists, p * 100)
        candidatos = celda_k[dists <= umbral].astype(C.dtype)

        if verbose:
            print(f"[DBS-{mode}] k={k}  z_k={zk}  |Celda_k|={len(celda_k)}  |N(z_k)|={len(candidatos)}")

        # ── Evaluación batched y aceptación estricta ─────────────────────

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

    # ── Se retornan todas las candidatas que mejoran ambos objetivos ──
    return aceptadas

#######################################################
# Núcleo común de run_nsga2_PNS / run_nsga2_DBS: ciclo de
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
    history_hv_f1 = [hypervolume_f1(objectives[initial_f1])] if len(objectives) else []

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
        f1_hv = hypervolume_f1(objectives[f1_indices])

        history_objectives.append(objectives.copy())
        history_labels.append(labels_pop.copy())
        history_hv_f1.append(f1_hv)

        print(
            f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  "
            f"población = {len(population)}  |  F1 = {len(f1_indices)}  |  "
            f"HV F1 = {f1_hv:.6f}"
        )

    print(f"\nCriterio de paro EXACTO: {solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_function_calls} llamadas a la función objetivo, {gen} generaciones completadas.")

    return {
        "population": population,
        "objectives": objectives,
        "labels": labels_pop,
        "fronts": non_dominated_sort(objectives),
        "history_objectives": history_objectives,
        "history_labels": history_labels,
        "history_hv_f1": history_hv_f1,
        "history_hv_f1_mean": [float(hv) for hv in history_hv_f1],
        "generations_run": gen,
    }

#######################################################
# run_nsga2_PNS: variante de run_nsga2 que agrega Pareto-Neighborhood
# Search (PNS) como estrategia de búsqueda local
#######################################################

def run_nsga2_PNS(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    nF: int = 1,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    pns_local_budget: int | None = None,
    seed: int | None = None,
) -> dict:
    """
    Variante de run_nsga2 que agrega PNS: tras construir Rv = Pv ∪ Qv, se
    aplica PNS (`nF` frentes de Pareto de UN medoide al azar) sobre cada solución de su F1; los reemplazos aceptados (mejoran al menos un Xie-Beni) se agregan
    a Rv antes de la selección de sobrevivientes. Ver _run_nsga2_BI.
    """
    def apply_operator(C, labels, objs, limit, rng, _population):
        return pareto_neighborhood_search(
            C, labels, ge_matrix, bi_matrix, nF=nF,
            max_obj_calls=limit, base_objectives=tuple(objs), rng=rng,
        )

    return _run_nsga2_BI(
        apply_operator, ge_matrix, bi_matrix, n, k, pop_size, max_obj_function_calls,
        crossover_prob, mutation_prob, pns_local_budget, seed,
    )

#######################################################
# run_nsga2_DBS: variante de run_nsga2 que agrega Discrepancy-Based Search
# (DBS, régimen `mode`) como estrategia de búsqueda local
#######################################################

def run_nsga2_DBS(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    mode: str = "bio",
    eps: float = DBS_EPS,
    p: float = 0.10,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    dbs_local_budget: int | None = None,
    seed: int | None = None,
) -> dict:
    """
    Variante de run_nsga2 que agrega DBS: tras construir Rv = Pv ∪ Qv, se
    aplica DBS sobre cada solución de su F1; los reemplazos aceptados (mejoran estrictamente ambos Xie-Beni) se agregan a Rv antes de la
    selección de sobrevivientes. Ver _run_nsga2_BI.

    mode : "bio" | "no_discrepancy" | "expresion" — fijo por corrida. D_mode
           no depende de la población, así que se construye UNA sola vez.
    eps  : ε de la matriz de discrepancia.
    p    : percentil de vecindad dentro de la celda de cada medoide.
    """
    assert mode in DBS_MODES, f"mode debe ser uno de {DBS_MODES}."
    d_mode = build_discrepancy_matrix(ge_matrix, bi_matrix, mode, eps)

    def apply_operator(C, labels, objs, limit, rng, _population):
        return discrepancy_based_search(
            C, labels, ge_matrix, bi_matrix, mode=mode, eps=eps, p=p, d_mode=d_mode,
            max_obj_calls=limit, base_objectives=tuple(objs), rng=rng,
        )

    return _run_nsga2_BI(
        apply_operator, ge_matrix, bi_matrix, n, k, pop_size, max_obj_function_calls,
        crossover_prob, mutation_prob, dbs_local_budget, seed,
    )
