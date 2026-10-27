"""
Operadores de búsqueda local (LS) desarrollados en Notes_2-LSOperators.ipynb:
Multi-Objective Path-Relinking (MOPR), Pareto Local Search (PLS) y
MOLS (variantes large / narrow). Se apoyan en solution_encoding y
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

############################
# Importaciones
############################
import numpy as np                  # Operaciones matemáticas eficientes.

# Importaciones propias.
import solution_encoding
from solution_encoding  import build_clusters_population, xie_beni, xie_beni_population
from pareto_sorting     import non_dominated_sort, crowding_distance, _dominates


############################
# Utilidades compartidas
############################

def evaluate_objectives(
    medoids: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
) -> tuple[float, float]:
    """
    Evalúa (XBGE, XBBI) para UNA única solución de medoides, envolviendo
    build_clusters_population + xie_beni (pensadas para poblaciones) para
    el caso de un solo individuo, que es lo que requieren MOPR y PLS.

    Nota
    ----
    Siguiendo la misma libertad de eficiencia adoptada en el NSGA-II
    (Notes_1), la asignación de clusters se calcula UNA vez con GE y se
    reutiliza para ambos índices Xie-Beni, en lugar de recalcularla también
    con BI como sugiere estrictamente la Tabla 1 del paper. Si se decide
    revertir esta libertad, basta con calcular una segunda asignación con
    build_clusters_population(bi_matrix, medoids[None, :])[0] y ajustar
    xie_beni para aceptar labels distintas por matriz.
    """
    labels = build_clusters_population(ge_matrix, medoids[None, :])[0]
    return xie_beni(ge_matrix, bi_matrix, medoids, labels)

def evaluate_objectives_population(
    medoids_pop: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
) -> np.ndarray:
    """
    Versión "batch" de evaluate_objectives: evalúa (XBEB, XBBB) para TODA
    una población de medoides de una sola vez, apilando build_clusters_population
    + xie_beni_population en vez de iterar evaluate_objectives candidata por
    candidata. Retorna un array (m, 2).

    Nota
    ----
    xie_beni_population incrementa solution_encoding.OBJ_FUNCTION_CALLS en
    `len(medoids_pop)` (una unidad por individuo evaluado, igual que llamar
    evaluate_objectives m veces), por lo que el criterio de detención por
    cuota de llamadas no cambia: el llamador debe seguir recortando
    `medoids_pop` al presupuesto restante ANTES de invocar esta función,
    tal como ya se hacía con las listas de candidatas evaluadas 1 a 1.
    """
    labels_pop = build_clusters_population(ge_matrix, medoids_pop)
    return xie_beni_population(ge_matrix, bi_matrix, medoids_pop, labels_pop)


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
    start_objective: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Construye la trayectoria PR(start, guide). Respeta `max_obj_calls`
    (misma cuota GLOBAL de nsga2.py, contra
    solution_encoding.OBJ_FUNCTION_CALLS): si el presupuesto restante no
    alcanza para evaluar todas las combinaciones candidatas de un paso, se
    recorta la lista de candidatas; si se agota antes de terminar, la
    trayectoria vuelve INCOMPLETA (no necesariamente llega a `guide`).

    Retorna (trajectory, trajectory_obj): cada solución de la trayectoria
    junto con su objetivo (XBEB, XBBB), ya calculado al elegir el mejor
    movimiento de cada paso — así no se vuelve a evaluar (ni a gastar
    cuota) después. `start_objective` es el objetivo ya conocido de
    `start`; si es None, se evalúa (1 llamada).
    """
    set_start = set(start.tolist())
    set_guide = set(guide.tolist())
    to_remove = sorted(set_start - set_guide)
    to_add = sorted(set_guide - set_start)
    assert len(to_remove) == len(to_add), (
        "El número de medoides a remover y agregar debe coincidir."
    )

    if start_objective is None:
        start_objective = evaluate_objectives_population(start[None, :], ge_matrix, bi_matrix)[0]

    current = start.copy()
    trajectory = [current.copy()]
    trajectory_obj = [np.asarray(start_objective, dtype=np.float64)]

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

        # Evaluación de los cambios realizados en esa iteración (batch único).
        objectives = evaluate_objectives_population(np.stack(candidates), ge_matrix, bi_matrix)

        # Seleccionar la solución con mejor rendimiento.
        best_idx = _select_best(objectives)
        current = candidates[best_idx]
        zk_chosen, zl_chosen = moves[best_idx]

        # Establecer como nueva posición de la trayectoria (con su objetivo).
        to_remove.remove(zk_chosen)
        to_add.remove(zl_chosen)
        trajectory.append(current.copy())
        trajectory_obj.append(np.asarray(objectives[best_idx], dtype=np.float64))

        if verbose:
            xbeb, xbbb = objectives[best_idx]
            print(
                f"[PR] swap z_{zk_chosen} → z_{zl_chosen}  → {current.tolist()}  "
                f"(XBEB={xbeb:.6f}, XBBB={xbbb:.6f})  [llamadas={solution_encoding.OBJ_FUNCTION_CALLS}/{max_obj_calls}]"
            )

    return trajectory, trajectory_obj

def multi_objective_path_relinking(
    C1: np.ndarray,
    C2: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    max_obj_calls: int,
    C1_objective: np.ndarray | None = None,
    C2_objective: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """
    Aplica MOPR completo sobre (C1, C2), respetando `max_obj_calls` en cada
    etapa: si la cuota se agota tras PR(C1,C2), se omite PR(C2,C1).

    El pool se arma con los objetivos que ya calculó path_relinking, por lo
    que NO se evalúa nada adicional (cada solución se paga una sola vez).
    `C1_objective` / `C2_objective` son los objetivos ya conocidos de C1 y
    C2 (p.ej. los del F1 de Rv); si son None, se evalúan.
    """
    k = len(C1)

    if solution_encoding.OBJ_FUNCTION_CALLS >= max_obj_calls:
        if verbose:
            print("[MOPR] Cuota global ya agotada antes de iniciar.")
        empty = np.empty((0, k), dtype=C1.dtype)
        return {"pool": empty, "pool_obj": np.empty((0, 2)), "f1": empty, "f1_obj": np.empty((0, 2)),
                "trajectory_C1_to_C2": [], "trajectory_C2_to_C1": []}

    if C1_objective is None or C2_objective is None:
        both = evaluate_objectives_population(np.stack([C1, C2]), ge_matrix, bi_matrix)
        C1_objective = both[0] if C1_objective is None else C1_objective
        C2_objective = both[1] if C2_objective is None else C2_objective

    if verbose:
        print("\n[MOPR] Trayectoria PR(C1, C2)")
    traj_c1_c2, obj_c1_c2 = path_relinking(
        C1, C2, ge_matrix, bi_matrix, max_obj_calls, start_objective=C1_objective, verbose=verbose)

    if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_calls:
        if verbose:
            print("\n[MOPR] Trayectoria PR(C2, C1)")
        traj_c2_c1, obj_c2_c1 = path_relinking(
            C2, C1, ge_matrix, bi_matrix, max_obj_calls, start_objective=C2_objective, verbose=verbose)
    else:
        traj_c2_c1, obj_c2_c1 = [], []
        if verbose:
            print("[MOPR] Cuota global agotada tras PR(C1,C2); se omite PR(C2,C1).")

    # Fusionar y eliminar duplicados por conjunto de medoides (conservando
    # el objetivo ya calculado de cada solución).
    unique_pool: dict[frozenset, tuple[np.ndarray, np.ndarray]] = {}
    for sol, obj in zip(traj_c1_c2 + traj_c2_c1, obj_c1_c2 + obj_c2_c1):
        key = frozenset(sol.tolist())
        if key not in unique_pool:
            unique_pool[key] = (sol, obj)
    pool = np.stack([s for s, _ in unique_pool.values()])
    pool_obj = np.stack([o for _, o in unique_pool.values()])

    fronts = non_dominated_sort(pool_obj)
    f1_indices = fronts[0]
    f1 = pool[f1_indices]
    f1_obj = pool_obj[f1_indices]

    if verbose:
        print(f"\n[MOPR] Pool: {pool.shape[0]} soluciones "
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

def generate_neighborhood(C, n, rng, forbidden, max_neighbors):
    """
    Generación de vecindario de una solución: se elige UNA posición de C al
    azar y se reemplaza su medoide por una submuestra aleatoria (sin
    reposición) de genes que no pertenecen a NINGUNA solución del archivo.

    Parámetros
    ----------
    forbidden     : np.ndarray — genes usados como medoide en todo el archivo.
    max_neighbors : int — tamaño máximo de la submuestra, ceil(neighborhood * n).

    Retorna
    -------
    neighbors : np.ndarray (m, k) — m = min(max_neighbors, genes disponibles).
    pos       : int — posición del cromosoma modificada.
    zk        : int — medoide original en esa posición.
    """
    k = len(C)
    pos = int(rng.integers(0, k))
    zk = int(C[pos])

    # Genes disponibles para reemplazo: todos menos los del archivo.
    available = np.ones(n, dtype=bool)
    available[forbidden] = False
    pool_z = np.flatnonzero(available)

    if pool_z.size == 0:
        return np.empty((0, k), dtype=C.dtype), pos, zk

    # Submuestra aleatoria sin reposición.
    n_target = min(max_neighbors, pool_z.size)
    picks = rng.choice(pool_z, size=n_target, replace=False)

    neighbors = np.tile(C, (n_target, 1))
    neighbors[:, pos] = picks
    return neighbors, pos, zk


def _feasible_mask(neighbors: np.ndarray, ge_matrix: np.ndarray) -> np.ndarray:
    """
    Máscara de factibilidad: una solución es factible si ningún cluster
    queda con menos de 2 elementos. No consume cuota de llamadas.
    """
    m, k = neighbors.shape
    labels = build_clusters_population(ge_matrix, neighbors)       # (m, n), 1-based
    flat = (labels - 1) + np.arange(m)[:, None] * k                # índice fila*k + cluster
    counts = np.bincount(flat.ravel(), minlength=m * k).reshape(m, k)
    return (counts >= 2).all(axis=1)


def pareto_local_search(
    initial_population: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    max_obj_calls: int,
    initial_objectives: np.ndarray | None = None,
    local_budget: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
    neighborhood: float = 1.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Aplica Pareto Local Search (PLS), respetando cuota GLOBAL + LOCAL.

    Parámetros
    ----------
    initial_objectives : np.ndarray | None — objetivos (XBEB, XBBB) YA
        CONOCIDOS para `initial_population`, en el mismo orden y shape
        (m, 2). Si se proveen, NO se vuelve a llamar evaluate_objectives
        sobre esas soluciones. Si es None, se evalúan todas desde cero,
        consumiendo len(initial_population) llamadas antes de explorar un
        solo vecino.
    neighborhood : float en (0, 1] — fracción de n que define el tamaño
        máximo de N(C): ceil(neighborhood * n) vecinos por exploración.
    """
    # Inicialización de elementos de números pseudoaleatorios.
    rng = np.random.default_rng(seed)
    n = ge_matrix.shape[0]
    max_neighbors = int(np.ceil(neighborhood * n))

    # Administración de llamadas restantes vs las permitidas por parametro.
    calls_start = solution_encoding.OBJ_FUNCTION_CALLS
    local_cap = (calls_start + local_budget) if local_budget is not None else max_obj_calls
    effective_max = min(max_obj_calls, local_cap)

    # Verificar existencia de valores en función obj.
    if initial_objectives is not None and len(initial_objectives) != len(initial_population):
        raise ValueError(
            f"'initial_objectives' ({len(initial_objectives)}) debe tener el mismo "
            f"largo que 'initial_population' ({len(initial_population)})."
        )

    # Evaluar o no conjunto de soluciones iniciales según el estado de entrada.
    pool: dict[frozenset, dict] = {}
    if initial_objectives is not None:
        for idx, sol in enumerate(initial_population):
            key = frozenset(sol.tolist())
            if key in pool:
                continue
            obj = tuple(initial_objectives[idx])
            pool[key] = {"solution": sol.copy(), "objectives": obj, "explored": False}
    else:
        to_eval_keys: list[frozenset] = []
        to_eval_sols: list[np.ndarray] = []
        seen_keys: set[frozenset] = set()
        for sol in initial_population:
            key = frozenset(sol.tolist())
            if key in pool or key in seen_keys:
                continue
            if solution_encoding.OBJ_FUNCTION_CALLS + len(to_eval_sols) >= effective_max:
                break
            seen_keys.add(key)
            to_eval_keys.append(key)
            to_eval_sols.append(sol)
        if to_eval_sols:
            objs = evaluate_objectives_population(np.stack(to_eval_sols), ge_matrix, bi_matrix)
            for key, sol, obj in zip(to_eval_keys, to_eval_sols, objs):
                pool[key] = {"solution": sol.copy(), "objectives": tuple(obj), "explored": False}

    if verbose:
        print(f"[PLS] Población inicial A0: {len(pool)} soluciones "
              f"(llamadas={solution_encoding.OBJ_FUNCTION_CALLS}/{effective_max}, "
              f"objetivos_precalculados={initial_objectives is not None}).")

    # Iteración según cuota de llamadas disponibles y establecidas por parametros.
    while True:
        # a. -- Revisión de llamadas en cuotas.
        if solution_encoding.OBJ_FUNCTION_CALLS >= effective_max:
            if verbose:
                reason = "cuota LOCAL de esta llamada" if effective_max < max_obj_calls else "cuota GLOBAL"
                print(f"[PLS] {reason} agotada ({solution_encoding.OBJ_FUNCTION_CALLS}/{effective_max}). Deteniendo.")
            break

        # b. -- Soluciones sin explorar.
        unexplored_keys = [key for key, entry in pool.items() if not entry["explored"]]
        if not unexplored_keys:
            if verbose:
                print(f"[PLS] A0 vacía. Población final |A|={len(pool)}.")
            break

        # c. -- Selección de solución a explorar al azar.
        c_key = unexplored_keys[int(rng.integers(0, len(unexplored_keys)))]
        C = pool[c_key]["solution"]
        C_obj = pool[c_key]["objectives"]

        # d. -- Vecindad: un medoide de C reemplazado por una submuestra de
        # genes que no pertenecen a ninguna solución del archivo.
        forbidden = np.unique(np.stack([e["solution"] for e in pool.values()]))
        neighbors, pos, zk = generate_neighborhood(C, n, rng, forbidden, max_neighbors)

        # Filtro de factibilidad (clusters con >= 2 elementos). Va ANTES de
        # recortar por presupuesto para no gastar llamadas en infactibles.
        if len(neighbors):
            neighbors = neighbors[_feasible_mask(neighbors, ge_matrix)]

        remaining = effective_max - solution_encoding.OBJ_FUNCTION_CALLS
        if remaining <= 0:
            break
        if len(neighbors) > remaining:
            neighbors = neighbors[:remaining]

        if verbose:
            print(f"[PLS] C={C.tolist()}  (XBEB={C_obj[0]:.4f}, XBBB={C_obj[1]:.4f})  "
                  f"reemplazando posición {pos} (z_{zk}), |N(C)|={len(neighbors)}")

        # e. -- Sin vecinos factibles: C queda explorada.
        if len(neighbors) == 0:
            pool[c_key]["explored"] = True
            continue

        # f. -- Criterio de aceptación: ordenamiento no dominado CONJUNTO sobre
        # vecinos + archivo; el nuevo archivo es el frente F1 de esa unión.
        neighbor_objs = evaluate_objectives_population(neighbors, ge_matrix, bi_matrix)

        keys = list(pool.keys())
        all_objs = [pool[key]["objectives"] for key in keys] + [tuple(o) for o in neighbor_objs]
        f1 = set(non_dominated_sort(all_objs)[0])

        new_pool: dict[frozenset, dict] = {}
        for i, key in enumerate(keys):
            if i in f1:
                new_pool[key] = pool[key]                      # hereda 'explored'
        for j, (neighbor, neighbor_obj) in enumerate(zip(neighbors, neighbor_objs)):
            if len(keys) + j in f1:
                new_pool[frozenset(neighbor.tolist())] = {
                    "solution": neighbor,
                    "objectives": tuple(neighbor_obj),
                    "explored": False,
                }
        pool = new_pool

        # C se marca explorada aunque haya sido dominada y eliminada.
        if c_key in pool:
            pool[c_key]["explored"] = True

    if not pool:
        k = initial_population.shape[1]
        return np.empty((0, k), dtype=initial_population.dtype), []

    population = np.stack([entry["solution"] for entry in pool.values()])
    objectives = [entry["objectives"] for entry in pool.values()]
    return population, objectives


#######################################################
# MODULO MOLS
# (cuota global+local, dominancia vectorizada, sin re-evaluación redundante)
#######################################################

def build_neighborhood_matrix(ge_matrix: np.ndarray, bi_matrix: np.ndarray, source: str = "BOTH") -> np.ndarray:
    """
    Matriz M_V sobre la que MOLS define el vecindario. `source` selecciona:
    "GE"   → solo expresión (ge_matrix).
    "BI"   → solo información biológica (bi_matrix).
    "BOTH" → combinación sqrt(GE² + BI²) (distancia euclidiana conjunta).
    """
    source = source.upper()
    assert source in ("GE", "BI", "BOTH"), "source debe ser 'GE', 'BI' o 'BOTH'."
    if source == "GE":
        return ge_matrix
    if source == "BI":
        return bi_matrix
    return np.sqrt(ge_matrix ** 2 + bi_matrix ** 2)

def generate_neighborhood_mols(
    C: np.ndarray,
    m_v_matrix: np.ndarray,
    neighborhood: float,
) -> list[np.ndarray]:
    """
    Neighborhood(C, M_V) según ecuación 3.4: un medoide reemplazado por
    un gen no presente con M_V(reemplazado, reemplazo) <= neighborhood.
    """
    k = len(C)
    bloques = []

    for pos in range(k):
        zk = int(C[pos])
        close_genes = np.where(m_v_matrix[zk] <= neighborhood)[0]

        # 'zk in C' ya cubre el caso zl == zk (zk siempre pertenece a C),
        # así que un solo np.isin reemplaza el "if zl == zk or zl in used".
        valid_zl = close_genes[~np.isin(close_genes, C)]
        if len(valid_zl) == 0:
            continue

        bloque = np.tile(C, (len(valid_zl), 1))   # (n_validos, k) — una sola asignación vectorizada
        bloque[:, pos] = valid_zl
        bloques.append(bloque)

    if not bloques:
        return []

    return list(np.vstack(bloques))

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

def mols(
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
    MOLS (Algoritmos 3.6 y 3.7, Toledo 2021). `mode` selecciona el criterio de aceptación:
    "large"  → acepta un vecino si la solución NO lo domina (no-dominación).
    "narrow" → acepta un vecino solo si DOMINA a la solución (dominación).

    Parámetros
    ----------
    m_v_matrix: np.ndarray - Considerar que puede ser la propia matriz de
        expresión o la de información biológica.
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
    # Comprobación de modo para aceptar soluciones.
    assert mode in ("large", "narrow"), "mode debe ser 'large' o 'narrow'."
    if population_objectives is not None and len(population_objectives) != len(population):
        raise ValueError(
            f"'population_objectives' ({len(population_objectives)}) debe tener el mismo "
            f"largo que 'population' ({len(population)})."
        )

    # Administración de llamadas a función objetivo.
    calls_start = solution_encoding.OBJ_FUNCTION_CALLS
    local_cap = (calls_start + local_budget) if local_budget is not None else max_obj_calls
    effective_max = min(max_obj_calls, local_cap)

    # Inicialización de conjuntos para aplicar MOLS.
    archive: list[np.ndarray] = []
    archive_obj: list[np.ndarray] = []
    seen = {frozenset(sol.tolist()) for sol in population}

    # Por cada solución en el conjunto de entrada se debe
    # asegurar que todas las soluciones tengan valores en
    # función objetivo.
    for idx, solution in enumerate(population):
        if population_objectives is not None:
            solution_obj = np.asarray(population_objectives[idx], dtype=np.float64)
        else:
            if solution_encoding.OBJ_FUNCTION_CALLS >= effective_max:
                break
            solution_obj = np.asarray(evaluate_objectives(solution, ge_matrix, bi_matrix), dtype=np.float64)

        # Generar vecindario según matriz Mv.
        neighbors = generate_neighborhood_mols(solution, m_v_matrix, neighborhood)
        neighbors = [nb for nb in neighbors if frozenset(nb.tolist()) not in seen]

        # Evaluar soluciones según cuota de llamadas disponibles.
        remaining = effective_max - solution_encoding.OBJ_FUNCTION_CALLS
        if remaining <= 0:
            break
        if len(neighbors) > remaining:
            neighbors = neighbors[:remaining]
        if not neighbors:
            continue
        neighbor_objs = evaluate_objectives_population(np.stack(neighbors), ge_matrix, bi_matrix)

        # Establecer condiciones para seleccionar soluciones aceptadas.
        if mode == "large":
            accept_mask = ~_dominates_vec(solution_obj, neighbor_objs)
        else:
            accept_mask = _is_dominated_by_vec(solution_obj, neighbor_objs)

        # Ingreso de nuevas soluciones al conjunto (sálida).
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
