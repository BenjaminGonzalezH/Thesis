"""
Operadores de búsqueda local (LS) desarrollados en Notes_2-LSOperators.ipynb:
Multi-Objective Path-Relinking (MOPR), Pareto Local Search (PLS) y
Large/Narrow-MOLS (L-MOLS / N-MOLS). Se apoyan en solution_encoding y
pareto_sorting, y son consumidos por nsga2_variants.py.

Referencias:
[1] Parraga-Alava, J., Dorn, M. & Inostroza-Ponta, M. A multi-objective gene
    clustering algorithm guided by a priori biological knowledge with
    intensification and diversification strategies. BioData Mining 11, 16
    (2018). https://doi.org/10.1186/s13040-018-0178-4
[2] Mariangel, N. (2020). Clustering multiobjetivo de datos de expresión
    génica de cáncer incorporando búsqueda local y conocimiento biológico a
    priori [Trabajo de titulación]. Universidad de Santiago de Chile (USACH).
"""

import numpy as np

import solution_encoding
from solution_encoding import build_clusters_population, xie_beni
from pareto_sorting import non_dominated_sort, crowding_distance


############################
# Utilidades compartidas
############################
def evaluate_objectives(
    medoids: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
) -> tuple[float, float]:
    """
    Evalúa (XBEB, XBBB) para UNA única solución de medoides, envolviendo
    build_clusters_population + xie_beni (pensadas para poblaciones) para
    el caso de un solo individuo, que es lo que requieren los operadores LS.

    Nota
    ----
    Siguiendo la misma libertad de eficiencia adoptada en el NSGA-II
    (Notes_1), la asignación de clusters se calcula UNA vez con GE y se
    reutiliza para ambos índices Xie-Beni, en lugar de recalcularla también
    con bi como sugiere estrictamente la Tabla 1 del paper. Si se decide
    revertir esta libertad, basta con calcular una segunda asignación con
    build_clusters_population(bi_matrix, medoids[None, :])[0] y ajustar
    xie_beni para aceptar labels distintas por matriz.
    """
    labels = build_clusters_population(ge_matrix, medoids[None, :])[0]
    return xie_beni(ge_matrix, bi_matrix, medoids, labels)


def _dominates(obj_a, obj_b) -> bool:
    """
    Determina si la solución A domina a la solución B (minimización),
    según la definición (2) del paper:

        A ≺ B  ⟺  ∀t: P_t(A) ≤ P_t(B)  ∧  ∃t: P_t(A) < P_t(B)

    Parámetros
    ----------
    obj_a, obj_b : secuencias de floats (ej. (XBEB, XBBB)).
    """
    not_worse = all(a <= b for a, b in zip(obj_a, obj_b))
    strictly_better = any(a < b for a, b in zip(obj_a, obj_b))
    return not_worse and strictly_better


#######################################################
# MODULO MULTI-OBJECTIVE PATH-RELINKING (MOPR)
# Se aplica el operador de búsqueda local Path Relinking de
# la forma señalada en el paper de Parraga, en el cual se
# establecen trayectorias en ambos sentidos.
#######################################################

def _select_best(objectives: list[tuple[float, float]]) -> int:
    """
    Selecciona el mejor índice de un conjunto de candidatas: primero por
    frente de Pareto (F1) y, en caso de empate, por la mayor crowding
    distance — el "best move" de cada paso de la trayectoria.
    """
    # Nota: Considerar que PR solo es utilizado en la F1 al tomar
    # las mejores soluciones resultantes de la trayectoria.
    fronts = non_dominated_sort(objectives)
    f1 = fronts[0]

    if len(f1) == 1:
        return f1[0]

    dist = crowding_distance(objectives, f1)
    return max(f1, key=lambda i: dist[i])


def path_relinking(
    start: np.ndarray,
    guide: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    max_obj_calls: int,
    verbose: bool = False,
) -> list[np.ndarray]:
    """
    Construye la trayectoria PR(start, guide). Respeta `max_obj_calls`
    (misma cuota GLOBAL de nsga2.py, contra
    solution_encoding.OBJ_FUNCTION_CALLS): si el presupuesto restante no
    alcanza para evaluar todas las combinaciones candidatas de un paso, se
    recorta la lista de candidatas; si se agota antes de terminar, la
    trayectoria vuelve INCOMPLETA (no necesariamente llega a `guide`).
    """
    set_start = set(start.tolist())
    set_guide = set(guide.tolist())
    to_remove = sorted(set_start - set_guide)
    to_add = sorted(set_guide - set_start)
    assert len(to_remove) == len(to_add), (
        "El número de medoides a remover y agregar debe coincidir."
    )

    current = start.copy()
    trajectory = [current.copy()]

    if verbose:
        print(f"[PR] start={start.tolist()}  guide={guide.tolist()}")
        print(f"[PR] to_remove={to_remove}  to_add={to_add}")

    while to_remove:
        # Corte GLOBAL antes de generar/evaluar más candidatas.
        if solution_encoding.OBJ_FUNCTION_CALLS >= max_obj_calls:
            if verbose:
                print(f"[PR] Cuota global agotada ({solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_calls}). "
                      f"Trayectoria incompleta ({len(to_remove)} intercambios pendientes).")
            break

        # Se establece iteración que para cada medoide a mover se
        # obtiene su respectivo indice, asimismo, se asegura de
        # probar todos los posibles según los candidatos disponibles.
        candidates, moves = [], []
        for zk in to_remove:
            pos = int(np.where(current == zk)[0][0])
            for zl in to_add:
                candidate = current.copy()
                candidate[pos] = zl
                candidates.append(candidate)
                moves.append((zk, zl))

        # Recortar candidatas al presupuesto GLOBAL restante.
        remaining = max_obj_calls - solution_encoding.OBJ_FUNCTION_CALLS
        if remaining <= 0:
            break
        if len(candidates) > remaining:
            candidates = candidates[:remaining]
            moves = moves[:remaining]

        # Evaluación de los cambios realizados en esa iteración.
        objectives = [evaluate_objectives(cand, ge_matrix, bi_matrix) for cand in candidates]

        # Seleccionar la solución con mejor rendimiento.
        best_idx = _select_best(objectives)
        current = candidates[best_idx]
        zk_chosen, zl_chosen = moves[best_idx]

        # Establecer como nueva posición de la trayectoria.
        to_remove.remove(zk_chosen)
        to_add.remove(zl_chosen)
        trajectory.append(current.copy())

        if verbose:
            xbeb, xbbb = objectives[best_idx]
            print(
                f"[PR] swap z_{zk_chosen} → z_{zl_chosen}  → {current.tolist()}  "
                f"(XBEB={xbeb:.6f}, XBBB={xbbb:.6f})  [llamadas={solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_calls}]"
            )

    return trajectory


def multi_objective_path_relinking(
    C1: np.ndarray,
    C2: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    max_obj_calls: int,
    verbose: bool = False,
) -> dict:
    """
    Aplica MOPR completo sobre (C1, C2), respetando `max_obj_calls` en cada
    etapa: si la cuota se agota tras PR(C1,C2), se omite PR(C2,C1); el pool
    final solo evalúa tantas soluciones como el presupuesto restante permita
    (si no queda nada, "f1" vuelve vacío).
    """
    k = len(C1)

    if solution_encoding.OBJ_FUNCTION_CALLS >= max_obj_calls:
        if verbose:
            print("[MOPR] Cuota global ya agotada antes de iniciar.")
        empty = np.empty((0, k), dtype=C1.dtype)
        return {"pool": empty, "pool_obj": [], "f1": empty, "f1_obj": [],
                "trajectory_C1_to_C2": [], "trajectory_C2_to_C1": []}

    if verbose:
        print("\n[MOPR] Trayectoria PR(C1, C2)")
    traj_c1_c2 = path_relinking(C1, C2, ge_matrix, bi_matrix, max_obj_calls, verbose=verbose)

    if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_calls:
        if verbose:
            print("\n[MOPR] Trayectoria PR(C2, C1)")
        traj_c2_c1 = path_relinking(C2, C1, ge_matrix, bi_matrix, max_obj_calls, verbose=verbose)
    else:
        traj_c2_c1 = []
        if verbose:
            print("[MOPR] Cuota global agotada tras PR(C1,C2); se omite PR(C2,C1).")

    # Fusionar y eliminar duplicados por conjunto de medoides.
    all_solutions = traj_c1_c2 + traj_c2_c1
    unique_pool: dict[frozenset, np.ndarray] = {}
    for sol in all_solutions:
        key = frozenset(sol.tolist())
        if key not in unique_pool:
            unique_pool[key] = sol
    pool = np.stack(list(unique_pool.values())) if unique_pool else np.empty((0, k), dtype=C1.dtype)

    # Evaluar el pool respetando lo que quede de presupuesto GLOBAL.
    remaining = max_obj_calls - solution_encoding.OBJ_FUNCTION_CALLS
    n_evaluable = max(0, min(len(pool), remaining))
    pool = pool[:n_evaluable]
    pool_obj = [evaluate_objectives(ind, ge_matrix, bi_matrix) for ind in pool]

    if len(pool) == 0:
        f1, f1_obj = pool, []
    else:
        fronts = non_dominated_sort(pool_obj)
        f1_indices = fronts[0]
        f1 = pool[f1_indices]
        f1_obj = [pool_obj[i] for i in f1_indices]

    if verbose:
        print(f"\n[MOPR] Pool evaluado: {pool.shape[0]} soluciones "
              f"(llamadas: {solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_calls}).")
        print(f"[MOPR] Frente no dominado (F1): {len(f1)} soluciones.")

    return {
        "pool": pool, "pool_obj": pool_obj,
        "f1": f1, "f1_obj": f1_obj,
        "trajectory_C1_to_C2": traj_c1_c2, "trajectory_C2_to_C1": traj_c2_c1,
    }


#######################################################
# MODULO PARETO LOCAL SEARCH (PLS)
#######################################################

def generate_neighborhood(
    C: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int, int]:
    """
    Genera el vecindario N(C) de una solución, reemplazando un medoide
    elegido al azar por cada elemento del dataset aún no presente en C.

    Retorna
    -------
    neighbors : list[np.ndarray] — |N(C)| = n - k vecinos.
    pos       : int — posición del cromosoma modificada.
    zk        : int — medoide original en esa posición.
    """
    k = len(C)                          # Número de medoides.
    pos = int(rng.integers(0, k))       # Se toma un elemento a reemplazar.
    zk = int(C[pos])                    # Medoide a reemplazar.

    # Medoides usados y lista de medoides posibles para remplazo.
    used = set(C.tolist())
    candidates_z = [z for z in range(n) if z not in used]

    # Generación de vecindario.
    neighbors = []
    for zl in candidates_z:
        neighbor = C.copy()
        neighbor[pos] = zl
        neighbors.append(neighbor)

    return neighbors, pos, zk


def pareto_local_search(
    initial_population: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    max_obj_calls: int,
    initial_objectives: np.ndarray | None = None,
    local_budget: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Aplica Pareto Local Search (PLS), respetando cuota GLOBAL + LOCAL.

    Parámetros
    ----------
    initial_objectives : np.ndarray | None — objetivos (XBEB, XBBB) YA
        CONOCIDOS para `initial_population`, en el mismo orden y shape
        (m, 2). Si se proveen, NO se vuelve a llamar evaluate_objectives
        sobre esas soluciones — evita gastar presupuesto re-confirmando
        algo que el llamador ya calculó (p.ej. el F1 de Rv en
        run_nsga2_PLS). Si es None, se evalúan todas desde cero,
        consumiendo len(initial_population) llamadas antes de explorar un
        solo vecino.
    """
    rng = np.random.default_rng(seed)
    n = ge_matrix.shape[0]

    calls_start = solution_encoding.OBJ_FUNCTION_CALLS
    local_cap = (calls_start + local_budget) if local_budget is not None else max_obj_calls
    effective_max = min(max_obj_calls, local_cap)

    if initial_objectives is not None and len(initial_objectives) != len(initial_population):
        raise ValueError(
            f"'initial_objectives' ({len(initial_objectives)}) debe tener el mismo "
            f"largo que 'initial_population' ({len(initial_population)})."
        )

    pool: dict[frozenset, dict] = {}
    for idx, sol in enumerate(initial_population):
        key = frozenset(sol.tolist())
        if key in pool:
            continue
        if initial_objectives is not None:
            # Objetivo YA CONOCIDO por el llamador: no se gasta presupuesto.
            obj = tuple(initial_objectives[idx])
        else:
            if solution_encoding.OBJ_FUNCTION_CALLS >= effective_max:
                break
            obj = evaluate_objectives(sol, ge_matrix, bi_matrix)
        pool[key] = {"solution": sol.copy(), "objectives": obj, "explored": False}

    if verbose:
        print(f"[PLS] Población inicial A0: {len(pool)} soluciones "
              f"(llamadas={solution_encoding.OBJ_FUNCTION_CALLS}/{effective_max}, "
              f"objetivos_precalculados={initial_objectives is not None}).")

    while True:
        if solution_encoding.OBJ_FUNCTION_CALLS >= effective_max:
            if verbose:
                reason = "cuota LOCAL de esta llamada" if effective_max < max_obj_calls else "cuota GLOBAL"
                print(f"[PLS] {reason} agotada ({solution_encoding.OBJ_FUNCTION_CALLS}/{effective_max}). Deteniendo.")
            break

        unexplored_keys = [key for key, entry in pool.items() if not entry["explored"]]
        if not unexplored_keys:
            if verbose:
                print(f"[PLS] A0 vacía. Población final |A|={len(pool)}.")
            break

        # ── 1. Selección ────────────────────────────────────────────────────
        c_key = unexplored_keys[int(rng.integers(0, len(unexplored_keys)))]
        C = pool[c_key]["solution"]
        C_obj = pool[c_key]["objectives"]

        # ── 2. Exploración de vecindad ──────────────────────────────────────
        neighbors, pos, zk = generate_neighborhood(C, n, rng)

        remaining = effective_max - solution_encoding.OBJ_FUNCTION_CALLS
        if remaining <= 0:
            break
        if len(neighbors) > remaining:
            neighbors = neighbors[:remaining]

        if verbose:
            print(f"[PLS] C={C.tolist()}  (XBEB={C_obj[0]:.4f}, XBBB={C_obj[1]:.4f})  "
                  f"reemplazando posición {pos} (z_{zk}), |N(C)|={len(neighbors)}")

        # ── 3. Criterio de aceptación (dominancia) ──────────────────────────
        for neighbor in neighbors:
            neighbor_key = frozenset(neighbor.tolist())
            neighbor_obj = evaluate_objectives(neighbor, ge_matrix, bi_matrix)

            if not _dominates(C_obj, neighbor_obj):
                if neighbor_key not in pool:
                    pool[neighbor_key] = {"solution": neighbor, "objectives": neighbor_obj, "explored": False}
                dominated_keys = [
                    key for key, entry in pool.items()
                    if key != neighbor_key and _dominates(neighbor_obj, entry["objectives"])
                ]
                for key in dominated_keys:
                    del pool[key]

        if c_key in pool:
            pool[c_key]["explored"] = True

    if not pool:
        k = initial_population.shape[1]
        return np.empty((0, k), dtype=initial_population.dtype), []

    population = np.stack([entry["solution"] for entry in pool.values()])
    objectives = [entry["objectives"] for entry in pool.values()]
    return population, objectives


#######################################################
# MODULO N-MOLS / L-MOLS
# (cuota global+local, dominancia vectorizada, sin re-evaluación redundante)
#######################################################

def build_neighborhood_matrix(ge_matrix: np.ndarray, bi_matrix: np.ndarray) -> np.ndarray:
    """Matriz de vecindario M_V (ecuación 3.3, Toledo 2021) = sqrt(GE² + bi²)."""
    return np.sqrt(ge_matrix ** 2 + bi_matrix ** 2)


def generate_neighborhood_mols(
    C: np.ndarray,
    m_v_matrix: np.ndarray,
    neighborhood: float,
) -> list[np.ndarray]:
    """Neighborhood(C, M_V) según ecuación 3.4: un medoide reemplazado por
    un gen no presente con M_V(reemplazado, reemplazo) <= neighborhood."""
    k = len(C)                  # Medoides.
    used = set(C.tolist())      # Medoides utilizados.
    neighbors = []              # Vecindario.

    # Por cada posición dentro del arreglo de medoides se verifican los
    # genes más cercanos al medoide a reemplazar según el parametro neighborhood.
    # Evitando que esta sea igual al gen que se esta cambiando y que ya
    # sea parte de la solución.
    for pos in range(k):
        zk = int(C[pos])
        close_genes = np.where(m_v_matrix[zk] <= neighborhood)[0]
        for zl in close_genes:
            zl = int(zl)
            if zl == zk or zl in used:
                continue
            neighbor = C.copy()
            neighbor[pos] = zl
            neighbors.append(neighbor)

    # Retorno de vecindario.
    return neighbors


def _dominates_vec(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """True donde 'a' domina a B[i] (minimización). a:(2,), B:(m,2)."""
    a = np.asarray(a, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    not_worse = np.all(a <= B, axis=1)
    strictly_better = np.any(a < B, axis=1)
    return not_worse & strictly_better


def _is_dominated_by_vec(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Inverso de _dominates_vec: True donde B[i] domina a 'a'."""
    a = np.asarray(a, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    not_worse = np.all(B <= a, axis=1)
    strictly_better = np.any(B < a, axis=1)
    return not_worse & strictly_better


def _mols_search(
    population: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    m_v_matrix: np.ndarray,
    neighborhood: float,
    max_obj_calls: int,
    mode: str,
    population_objectives: np.ndarray | None = None,
    local_budget: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Núcleo compartido de L-MOLS y N-MOLS (Algoritmos 3.6 y 3.7, Toledo 2021).

    Parámetros
    ----------
    max_obj_calls : int — cuota GLOBAL del run completo (contra
        solution_encoding.OBJ_FUNCTION_CALLS).
    population_objectives : np.ndarray | None — objetivos (XBEB, XBBB) YA
        CONOCIDOS para `population`, en el mismo orden (shape (m,2)). Evita
        re-evaluar (y re-gastar presupuesto) soluciones cuyo objetivo el
        llamador ya calculó (p.ej. el F1 de Rv en run_nsga2_MOLS).
    local_budget : int | None — cuota propia de ESTA llamada (análogo a
        pareto_local_search); evita que una invocación dentro del ciclo de
        NSGA-II consuma TODO el presupuesto restante de una vez.
    """
    assert mode in ("l_mols", "n_mols"), "mode debe ser 'l_mols' o 'n_mols'."
    if population_objectives is not None and len(population_objectives) != len(population):
        raise ValueError(
            f"'population_objectives' ({len(population_objectives)}) debe tener el mismo "
            f"largo que 'population' ({len(population)})."
        )

    calls_start = solution_encoding.OBJ_FUNCTION_CALLS
    local_cap = (calls_start + local_budget) if local_budget is not None else max_obj_calls
    effective_max = min(max_obj_calls, local_cap)

    archive: list[np.ndarray] = []
    archive_obj: list[np.ndarray] = []
    seen = {frozenset(sol.tolist()) for sol in population}

    for idx, solution in enumerate(population):
        if population_objectives is not None:
            solution_obj = np.asarray(population_objectives[idx], dtype=np.float64)
        else:
            if solution_encoding.OBJ_FUNCTION_CALLS >= effective_max:
                break
            solution_obj = np.asarray(evaluate_objectives(solution, ge_matrix, bi_matrix), dtype=np.float64)

        neighbors = generate_neighborhood_mols(solution, m_v_matrix, neighborhood)
        neighbors = [nb for nb in neighbors if frozenset(nb.tolist()) not in seen]

        remaining = effective_max - solution_encoding.OBJ_FUNCTION_CALLS
        if remaining <= 0:
            break
        if len(neighbors) > remaining:
            neighbors = neighbors[:remaining]
        if not neighbors:
            continue

        neighbor_objs = np.array([evaluate_objectives(nb, ge_matrix, bi_matrix) for nb in neighbors])

        if mode == "l_mols":
            accept_mask = ~_dominates_vec(solution_obj, neighbor_objs)
        else:
            accept_mask = _is_dominated_by_vec(solution_obj, neighbor_objs)

        for nb, nb_obj, accepted in zip(neighbors, neighbor_objs, accept_mask):
            if accepted:
                key = frozenset(nb.tolist())
                if key not in seen:
                    archive.append(nb)
                    archive_obj.append(nb_obj)
                    seen.add(key)

        if verbose:
            print(f"[MOLS] solución {solution.tolist()} → archive acumulado: {len(archive)} "
                  f"(llamadas: {solution_encoding.OBJ_FUNCTION_CALLS}/{effective_max})")

    if not archive:
        return np.empty((0, population.shape[1]), dtype=population.dtype), np.empty((0, 2))

    return np.stack(archive), np.asarray(archive_obj, dtype=np.float64)


def l_mols(
    population: np.ndarray, ge_matrix: np.ndarray, bi_matrix: np.ndarray,
    m_v_matrix: np.ndarray, neighborhood: float, max_obj_calls: int,
    population_objectives: np.ndarray | None = None,
    local_budget: int | None = None, verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Large-MOLS (Algoritmo 3.6): acepta un vecino si la solución NO lo domina (no-dominación)."""
    return _mols_search(population, ge_matrix, bi_matrix, m_v_matrix, neighborhood,
                         max_obj_calls, mode="l_mols", population_objectives=population_objectives,
                         local_budget=local_budget, verbose=verbose)


def n_mols(
    population: np.ndarray, ge_matrix: np.ndarray, bi_matrix: np.ndarray,
    m_v_matrix: np.ndarray, neighborhood: float, max_obj_calls: int,
    population_objectives: np.ndarray | None = None,
    local_budget: int | None = None, verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Narrow-MOLS (Algoritmo 3.7): acepta un vecino solo si DOMINA a la solución (dominación)."""
    return _mols_search(population, ge_matrix, bi_matrix, m_v_matrix, neighborhood,
                         max_obj_calls, mode="n_mols", population_objectives=population_objectives,
                         local_budget=local_budget, verbose=verbose)
