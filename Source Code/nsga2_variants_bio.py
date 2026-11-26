"""
Variantes de run_nsga2 desarrolladas en Notes_3-BIOperators.ipynb, que
agregan cada uno de los operadores de búsqueda local basados en información
biológica como estrategia aplicada sobre el frente no dominado F0 de
Rv = Pv ∪ Qv, antes de la selección de sobrevivientes.

- run_nsga2_PNS : agrega Pareto-Neighborhood Search (escape por estancamiento: azar).
- run_nsga2_DBS : agrega Discrepancy-Based Search (regímenes bio / no_discrepancy / expresion;
                   escape por estancamiento: azar).
- run_nsga2_MPS : agrega Module-Participation Search (módulos de D_BI sub-representados en F0;
                   el operador aplica su propia aceptación y retorna solo el frente de lo aceptado;
                   escape por estancamiento: azar).

Cada variante implementa su PROPIO NSGA-II (ciclo completo, sin interfaz
compartida). En PNS y DBS, el operador se aplica una vez sobre cada solución de F0
dentro de una misma generación (MPS se invoca una vez sobre todo F0). El resto del
ciclo (selección por torneo, crossover, mutación, truncamiento por
rank+crowding) es IDÉNTICO a run_nsga2 (nsga2.py). Ambas comparten la MISMA
cuota global (solution_encoding.OBJ_FUNCTION_CALLS vs max_obj_function_calls)
con los operadores, por lo que nunca se excede max_obj_function_calls.

Este módulo también contiene los operadores PNS, DBS y MPS (con tope de
cuota global opcional `max_obj_calls`), ya que las variantes dependen de ellos.
"""

import weakref

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster          # Agrupamiento jerárquico y corte del dendrograma (MPS).
from scipy.spatial.distance import squareform                  # Matriz cuadrada → forma condensada (MPS).

import solution_encoding
from solution_encoding import random_medoids_pop, build_clusters_population, xie_beni, xie_beni_population
from pareto_sorting import non_dominated_sort
from operators import binary_tournament_selection, k_point_crossover, controller_random_mutation
from performance import hypervolume_f1
from nsga2 import evaluate_population, compute_ranks_and_crowding, _ensure_unique

#######################################################
# MODULO PARETO-NEIGHBORHOOD SEARCH (PNS)
# Para UN medoide z_k de la solución (al azar) se explora su vecindario Pareto:
# los primeros nF frentes de no dominancia, sobre todo el dataset (sin los
# medoides de C), respecto a la cercanía (D_GE, D_BI) a z_k. Cada candidata
# reemplaza a z_k, la partición se reconstruye GLOBALMENTE y se acepta si
# se evalúa. La aceptación (F0 de archivo ∪ candidatas, como PLS) la realiza el NSGA-II de cada variante.
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
    de los primeros `nF` frentes de Pareto F_0 ∪ ... ∪ F_{nF-1}.

    excluir_mask : np.ndarray (n,) bool — True en los medoides de C.
    """
    mask = ~excluir_mask
    mask[anchor] = False
    elegibles = np.where(mask)[0]
    f_ge, f_bi = ge_matrix[anchor, elegibles], bi_matrix[anchor, elegibles]
    sel = _first_nF_fronts_2d(f_ge, f_bi, nF)
    return set(elegibles[sel].tolist())

def pareto_neighborhood_search(
    C: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    nF: int = 1,
    verbose: bool = False,
    max_obj_calls: int | None = None,
    rng: np.random.Generator | None = None,
    azar: bool = False,
) -> list[dict]:
    """
    Pareto-Neighborhood Search (PNS): se elige UN medoide z_k de C al azar
    (análogo a PLS) y se evalúa su vecindario Pareto (primeros `nF` frentes
    respecto a (D_GE, D_BI) a z_k, excluyendo todos los medoides de C). Cada
    candidata c reemplaza a z_k y se reconstruye la partición completa.

    El operador retorna TODAS las candidatas evaluadas; la aceptación (el
    nuevo archivo es el F0 de archivo ∪ candidatas, como en PLS) la realiza
    el NSGA-II que lo invoca.

    Parámetros
    ----------
    C          : np.ndarray (k,) — medoides de la solución actual.
    ge_matrix, bi_matrix : matrices de distancia n×n, YA normalizadas en [0,1].
    nF         : número de frentes de Pareto que se conservan por medoide
                 (nF=1 → solo genes no dominados respecto a su cercanía
                 (D_GE, D_BI) a z_k).

    max_obj_calls   : int | None — tope sobre solution_encoding.OBJ_FUNCTION_CALLS
        (cuota GLOBAL de nsga2.py). Si es None no hay tope. Si el
        presupuesto no alcanza para el batch completo, éste se recorta
        (al azar, con `rng`).
    rng             : np.random.Generator | None — selección de z_k y recorte.
    azar            : bool — si True (el algoritmo está estancado), el vecindario NO
        usa frentes de Pareto: es una muestra aleatoria sin reposición de genes
        fuera de C y fuera de los primeros nF frentes de z_k (aumenta la
        variabilidad de genes). Su tamaño es el del vecindario Pareto normal de
        z_k (mismo gasto de cuota que PNS sin estancamiento).

    Retorna
    -------
    list[dict] con llaves "solution", "labels", "xb_ge", "xb_bi" — una
    entrada por cada candidata evaluada (puede quedar vacía).

    Nota de rendimiento
    --------------------
    Las candidatas C_prime de z_k se apilan en un solo batch y se evalúan
    con UNA llamada a build_clusters_population + UNA llamada a
    xie_beni_population, en vez de candidata por candidata.
    """
    K = len(C)
    excluir_mask = np.zeros(ge_matrix.shape[0], dtype=bool)
    excluir_mask[C] = True                                   # TODOS los medoides actuales de la solución.

    if rng is None:
        rng = np.random.default_rng()

    # ── Un único medoide z_k elegido al azar (como PLS) ──
    k = int(rng.integers(1, K + 1))
    zk = int(C[k - 1])

    # ── Vecindario Pareto de z_k sobre TODO el dataset, excluyendo C ──
    if azar:
        # Mismo tamaño que el vecindario Pareto normal (nF frentes), pero con genes al azar
        # que NO pertenecen a esos frentes (ni a C): explora genes distintos a los habituales.
        N_pareto = _pareto_neighborhood(zk, ge_matrix, bi_matrix, excluir_mask, nF)
        elegibles = np.setdiff1d(np.flatnonzero(~excluir_mask), np.fromiter(N_pareto, dtype=np.int64))
        N_zk = set(rng.choice(elegibles, size=min(len(N_pareto), len(elegibles)), replace=False).tolist())
    else:
        N_zk = _pareto_neighborhood(zk, ge_matrix, bi_matrix, excluir_mask, nF)

    if verbose:
        print(f"[PNS] k={k}  z_k={zk}  |N(z_k)|={len(N_zk)}" + ("  (azar)" if azar else ""))

    if not N_zk:
        return []

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

    return [
        {
            "solution": C_primes[i].copy(),
            "labels": labels_primes[i],
            "xb_ge": float(xb_primes[i, 0]),
            "xb_bi": float(xb_primes[i, 1]),
        }
        for i in range(len(C_primes))
    ]

#######################################################
# MODULO DISCREPANCY-BASED SEARCH (DBS)
# Se construye una matriz de discrepancia logarítmica
#     L = log((D_BI + ε) / (D_GE + ε))
# y, según el régimen `mode`, se transforma en una matriz D_mode sobre la
# cual se reutiliza build_clusters_population para particionar los genes
# en "celdas" (medoide más cercano bajo D_mode). El vecindario de un
# medoide z_k elegido al azar es el percentil `p` de su celda (sin medoides
# de la solución) MENOS discrepante con z_k (menor |D_mode| = |L|). La
# partición se reconstruye globalmente con D_GE y el operador retorna TODAS las
# candidatas evaluadas (sin filtrar por objetivos, como PNS); la aceptación (F0
# de archivo ∪ candidatas, como PLS) la realiza el NSGA-II de cada variante.
# Si el algoritmo está estancado (DBS no aporta soluciones nuevas a Rv), el
# vecindario se reemplaza por una muestra aleatoria de genes fuera de C y de
# N(z_k), del mismo tamaño que N(z_k) (azar → más variabilidad de genes).
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
    Construye D_mode a partir de L = log((D_BI + ε) / (D_GE + ε)):

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
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    mode: str = "bio",
    eps: float = DBS_EPS,
    p: float = 0.10,
    d_mode: np.ndarray | None = None,
    verbose: bool = False,
    max_obj_calls: int | None = None,
    rng: np.random.Generator | None = None,
    azar: bool = False,
) -> list[dict]:
    """
    Discrepancy-Based Search (DBS): cada gen se asigna al medoide más cercano
    bajo D_mode (celdas). Se elige UN medoide z_k de C al azar (como PNS);
    Celda_k = genes de su celda sin los medoides de C, y su vecindario N(z_k)
    es el percentil `p` de Celda_k menos discrepante con z_k (menor |D_mode|). Cada
    candidato c ∈ N(z_k) reemplaza a z_k y la partición se reconstruye
    globalmente con D_GE.

    El operador retorna TODAS las candidatas evaluadas; la aceptación (el
    nuevo archivo es el F0 de archivo ∪ candidatas, como en PLS) la realiza
    el NSGA-II que lo invoca.

    Parámetros
    ----------
    C          : np.ndarray (k,) — medoides de la solución actual.
    ge_matrix, bi_matrix : matrices de distancia n×n, YA normalizadas en [0,1].
    mode       : "bio" | "no_discrepancy" | "expresion" (fijo por corrida).
    eps        : ε de la matriz de discrepancia (evita división por 0 y log 0).
    p          : percentil de vecindad dentro de la celda del medoide
        (ej. 0.10 → 10% menos discrepante con z_k).
    d_mode     : np.ndarray | None — D_mode precalculada; si es None se
        construye aquí. Conviene precalcularla al aplicar el operador
        sobre varias soluciones, ya que no depende de C.

    max_obj_calls   : int | None — tope sobre solution_encoding.OBJ_FUNCTION_CALLS
        (cuota GLOBAL de nsga2.py). Si es None no hay tope. Si el
        presupuesto no alcanza para el batch completo, éste se recorta
        (al azar, con `rng`).
    rng             : np.random.Generator | None — selección de z_k y recorte.
    azar            : bool — si True (el algoritmo está estancado), el vecindario es una
        muestra aleatoria sin reposición de genes de TODO el dataset fuera de C y
        de N(z_k) (aumenta la variabilidad de genes). Su tamaño es |N(z_k)| (mismo
        gasto de cuota que sin estancamiento).

    Retorna
    -------
    list[dict] con llaves "solution", "labels", "xb_ge", "xb_bi" — TODAS las
    candidatas evaluadas (sin filtrar por objetivos; puede quedar vacía).

    Nota de rendimiento
    --------------------
    Igual que en PNS: las candidatas C_prime de z_k se apilan en un solo
    batch y se evalúan con UNA llamada a build_clusters_population + UNA
    llamada a xie_beni_population. Además, Celda_k se filtra con una máscara
    booleana (vectorizada) en vez de un list comprehension de Python.
    """
    if d_mode is None:
        d_mode = build_discrepancy_matrix(ge_matrix, bi_matrix, mode, eps)

    K = len(C)
    es_medoide = np.zeros(ge_matrix.shape[0], dtype=bool)
    es_medoide[C] = True                                     # TODOS los medoides actuales de la solución.

    if rng is None:
        rng = np.random.default_rng()

    # Asignación de cada gen a su medoide más cercano bajo D_mode.
    labels_mode = build_clusters_population(d_mode, C[None, :])[0]

    # ── Un único medoide z_k elegido al azar (como PNS / PLS) ──
    k = int(rng.integers(1, K + 1))
    zk = int(C[k - 1])
    celda_k = np.where((labels_mode == k) & ~es_medoide)[0]       # Celda completa, sin medoides.

    if len(celda_k) == 0:
        if verbose:
            print(f"[DBS-{mode}] k={k}  z_k={zk}  celda vacía")
        return []

    # ── Vecindario: percentil p MENOS discrepante con z_k (menor |D_mode| = |L|) DENTRO de la celda ──
    # |D_mode| = |L| en los tres modos (L, -L, |L|).
    dists = np.abs(d_mode[zk, celda_k])
    umbral = np.percentile(dists, p * 100)
    candidatos = celda_k[dists <= umbral].astype(C.dtype)

    if azar:
        # Estancamiento: mismo tamaño que N(z_k), pero genes al azar de todo el dataset
        # que NO pertenecen a N(z_k) ni a C.
        elegibles = np.setdiff1d(np.flatnonzero(~es_medoide), candidatos)
        candidatos = rng.choice(
            elegibles, size=min(len(candidatos), len(elegibles)), replace=False,
        ).astype(C.dtype)

    if verbose:
        print(f"[DBS-{mode}] k={k}  z_k={zk}  |Celda_k|={len(celda_k)}  |N(z_k)|={len(candidatos)}"
              + ("  (azar)" if azar else ""))

    # ── Cuota global: recorte del batch ───────────────────────────────
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

    resultado = [
        {
            "solution": C_primes[i].copy(),
            "labels": labels_primes[i],
            "xb_ge": float(xb_primes[i, 0]),
            "xb_bi": float(xb_primes[i, 1]),
        }
        for i in range(len(C_primes))
    ]

    return resultado

#######################################################
# run_nsga2_PNS: NSGA-II propio que agrega Pareto-Neighborhood Search
# (PNS) como estrategia de búsqueda local
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
    seed: int | None = None,
    gens_estancamiento: int = 1,
) -> dict:
    """
    NSGA-II con PNS: tras construir Rv = Pv ∪ Qv se extrae su frente no
    dominado F0, que parte como ARCHIVO (aceptación tipo PLS). Cada solución
    del archivo (en orden aleatorio) recibe UNA aplicación de PNS
    dentro de la generación; cada una explora el vecindario Pareto (`nF`
    frentes) de UN medoide al azar. Tras evaluar las candidatas, el nuevo
    archivo es el F0 de (archivo ∪ candidatas). Las candidatas que entran al
    archivo no se vuelven a explorar en la generación, y las soluciones que
    salen del archivo (dominadas) tampoco. Al terminar, las soluciones NUEVAS
    del archivo (que no estaban en Rv) se agregan a Rv antes de la selección de
    sobrevivientes. Lo demás (torneo, crossover, mutación, truncamiento por
    rank+crowding) es IDÉNTICO a run_nsga2.

    No hay cuota local por generación: la búsqueda local queda acotada por
    una aplicación por solución de F0 y por la cuota global (solution_encoding.OBJ_FUNCTION_CALLS
    vs max_obj_function_calls), por lo que el criterio de paro es EXACTO.

    Estancamiento: la generación en que PNS no aporta soluciones nuevas a Rv
    cuenta como la 1ª; al llegar a `gens_estancamiento` el vecindario pasa a ser una
    muestra aleatoria de genes fuera de C y de los nF frentes de z_k, del mismo
    tamaño que el vecindario Pareto normal (azar → más variabilidad de genes),
    hasta volver a aportar.
    """
    rng = np.random.default_rng(seed)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, ge_matrix, bi_matrix, max_obj_function_calls=max_obj_function_calls,
    )

    initial_fronts = non_dominated_sort(objectives) if len(objectives) else [[]]
    initial_f0 = initial_fronts[0]

    history_objectives = [objectives.copy()]
    history_labels = [labels_pop.copy()]
    history_hv_f1 = [hypervolume_f1(objectives[initial_f0])] if len(objectives) else []

    gen = 0
    gens_sin_cambio = 0                                      # Generaciones seguidas en estancamiento (la que lo suscita cuenta como 1).
    estancado = False
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

        # ── Operador PNS: ÚNICO agregado respecto a run_nsga2 ────────────────
        if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls:
            f0_idx = non_dominated_sort(combined_objectives)[0]
            rv_keys = {frozenset(sol.tolist()) for sol in combined_population}

            # Archivo inicial = F0 de Rv: llave → (solución, objetivos, labels).
            archivo = {
                frozenset(combined_population[j].tolist()):
                    (combined_population[j], combined_objectives[j], combined_labels[j])
                for j in f0_idx
            }

            # Una aplicación por cada solución del F0 original, en orden aleatorio.
            for j in rng.permutation(f0_idx):
                clave_j = frozenset(combined_population[j].tolist())
                if solution_encoding.OBJ_FUNCTION_CALLS >= max_obj_function_calls:
                    break
                if clave_j not in archivo:
                    continue                              # Ya dominada por una candidata previa.

                candidatas = pareto_neighborhood_search(
                    combined_population[j], ge_matrix, bi_matrix, nF=nF,
                    max_obj_calls=max_obj_function_calls, rng=rng,
                    azar=estancado,                      # Estancado → muestra aleatoria (fuera de C y de los nF frentes).
                )

                # Solo candidatas que no estén en Rv ni en el archivo.
                nuevas = []
                for d in candidatas:
                    key = frozenset(d["solution"].tolist())
                    if key not in rv_keys and key not in archivo:
                        nuevas.append((key, d))
                if not nuevas:
                    continue

                # Criterio de aceptación (como PLS): F0 de archivo ∪ candidatas.
                keys = list(archivo.keys())
                objs = np.array(
                    [archivo[key][1] for key in keys] + [(d["xb_ge"], d["xb_bi"]) for _, d in nuevas],
                    dtype=np.float64,
                )
                f0 = set(non_dominated_sort(objs)[0])

                nuevo_archivo = {key: archivo[key] for i, key in enumerate(keys) if i in f0}
                for i, (key, d) in enumerate(nuevas):
                    if len(keys) + i in f0:
                        nuevo_archivo[key] = (d["solution"], (d["xb_ge"], d["xb_bi"]), d["labels"])
                archivo = nuevo_archivo

            # Solo soluciones REALMENTE nuevas (que no estaban en Rv).
            agregadas = [entry for key, entry in archivo.items() if key not in rv_keys]
            # Estancamiento: PNS no varió Rv (la generación en que ocurre cuenta como la 1ª).
            gens_sin_cambio = 0 if agregadas else gens_sin_cambio + 1
            estancado = gens_sin_cambio >= gens_estancamiento

            if agregadas:
                # Objetivos y labels YA calculados dentro del operador (no
                # re-evaluar: duplicaría el gasto de cuota).
                combined_population = np.vstack([combined_population, np.asarray([e[0] for e in agregadas], dtype=combined_population.dtype)])
                combined_objectives = np.vstack([combined_objectives, np.asarray([e[1] for e in agregadas], dtype=np.float64)])
                combined_labels = np.vstack([combined_labels, np.asarray([e[2] for e in agregadas], dtype=combined_labels.dtype)])

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
        f0_indices = current_fronts[0]
        f0_hv = hypervolume_f1(objectives[f0_indices])
        history_objectives.append(objectives.copy())
        history_labels.append(labels_pop.copy())
        history_hv_f1.append(f0_hv)

        print(
            f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  "
            f"población = {len(population)}  |  F0 = {len(f0_indices)}  |  "
            f"HV F0 = {f0_hv:.6f}"
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
# run_nsga2_DBS: NSGA-II propio que agrega Discrepancy-Based Search
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
    seed: int | None = None,
    gens_estancamiento: int = 1,
) -> dict:
    """
    NSGA-II con DBS: tras construir Rv = Pv ∪ Qv se extrae su frente no
    dominado F0, que parte como ARCHIVO (aceptación tipo PLS). Cada solución
    del archivo (en orden aleatorio) recibe UNA aplicación de DBS
    dentro de la generación; cada una explora el vecindario de UN medoide al
    azar. Tras evaluar las candidatas, el nuevo archivo es el F0 de (archivo ∪
    candidatas). Las candidatas que entran al archivo no se vuelven a explorar
    en la generación, y las soluciones que salen del archivo (dominadas)
    tampoco. Al terminar, las soluciones NUEVAS del archivo (que no estaban en
    Rv) se agregan a Rv antes de la selección de sobrevivientes. Lo demás
    (torneo, crossover, mutación, truncamiento por rank+crowding) es IDÉNTICO a
    run_nsga2.

    mode : "bio" | "no_discrepancy" | "expresion" — fijo por corrida; define la
           partición en celdas. D_mode no depende de la población, así que se
           construye UNA sola vez.
    eps  : ε de la matriz de discrepancia.
    p    : percentil de vecindad dentro de la celda de cada medoide.

    El vecindario es el percentil p MENOS discrepante con z_k y el operador
    retorna todas las candidatas evaluadas (la aceptación es la del archivo, como
    PNS). Estancamiento: la generación en que DBS no aporta soluciones nuevas a Rv
    cuenta como la 1ª; al llegar a `gens_estancamiento` el vecindario pasa a ser una
    muestra aleatoria de genes fuera de C y de N(z_k), del mismo tamaño que N(z_k),
    hasta volver a aportar.

    No hay cuota local por generación: la búsqueda local queda acotada por
    una aplicación por solución de F0 y por la cuota global (criterio de paro EXACTO).
    """
    assert mode in DBS_MODES, f"mode debe ser uno de {DBS_MODES}."
    d_mode = build_discrepancy_matrix(ge_matrix, bi_matrix, mode, eps)

    rng = np.random.default_rng(seed)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, ge_matrix, bi_matrix, max_obj_function_calls=max_obj_function_calls,
    )

    initial_fronts = non_dominated_sort(objectives) if len(objectives) else [[]]
    initial_f0 = initial_fronts[0]

    history_objectives = [objectives.copy()]
    history_labels = [labels_pop.copy()]
    history_hv_f1 = [hypervolume_f1(objectives[initial_f0])] if len(objectives) else []

    gen = 0
    gens_sin_cambio = 0                                      # Generaciones seguidas en estancamiento (la que lo suscita cuenta como 1).
    estancado = False
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

        # ── Operador DBS: ÚNICO agregado respecto a run_nsga2 ────────────────
        if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls:
            f0_idx = non_dominated_sort(combined_objectives)[0]
            rv_keys = {frozenset(sol.tolist()) for sol in combined_population}

            # Archivo inicial = F0 de Rv: llave → (solución, objetivos, labels).
            archivo = {
                frozenset(combined_population[j].tolist()):
                    (combined_population[j], combined_objectives[j], combined_labels[j])
                for j in f0_idx
            }

            # Una aplicación por cada solución del F0 original, en orden aleatorio.
            for j in rng.permutation(f0_idx):
                clave_j = frozenset(combined_population[j].tolist())
                if solution_encoding.OBJ_FUNCTION_CALLS >= max_obj_function_calls:
                    break
                if clave_j not in archivo:
                    continue                              # Ya dominada por una candidata previa.

                candidatas = discrepancy_based_search(
                    combined_population[j], ge_matrix, bi_matrix, mode=mode, eps=eps, p=p,
                    d_mode=d_mode, max_obj_calls=max_obj_function_calls, rng=rng,
                    azar=estancado,                      # Estancado → muestra aleatoria (fuera de C y de N(z_k)).
                )

                # Solo candidatas que no estén en Rv ni en el archivo.
                nuevas = []
                for d in candidatas:
                    key = frozenset(d["solution"].tolist())
                    if key not in rv_keys and key not in archivo:
                        nuevas.append((key, d))
                if not nuevas:
                    continue

                # Criterio de aceptación (como PLS): F0 de archivo ∪ candidatas.
                keys = list(archivo.keys())
                objs = np.array(
                    [archivo[key][1] for key in keys] + [(d["xb_ge"], d["xb_bi"]) for _, d in nuevas],
                    dtype=np.float64,
                )
                f0 = set(non_dominated_sort(objs)[0])

                nuevo_archivo = {key: archivo[key] for i, key in enumerate(keys) if i in f0}
                for i, (key, d) in enumerate(nuevas):
                    if len(keys) + i in f0:
                        nuevo_archivo[key] = (d["solution"], (d["xb_ge"], d["xb_bi"]), d["labels"])
                archivo = nuevo_archivo

            # Solo soluciones REALMENTE nuevas (que no estaban en Rv).
            agregadas = [entry for key, entry in archivo.items() if key not in rv_keys]
            # Estancamiento: DBS no varió Rv (la generación en que ocurre cuenta como la 1ª).
            gens_sin_cambio = 0 if agregadas else gens_sin_cambio + 1
            estancado = gens_sin_cambio >= gens_estancamiento
            if agregadas:
                # Objetivos y labels YA calculados dentro del operador (no
                # re-evaluar: duplicaría el gasto de cuota).
                combined_population = np.vstack([combined_population, np.asarray([e[0] for e in agregadas], dtype=combined_population.dtype)])
                combined_objectives = np.vstack([combined_objectives, np.asarray([e[1] for e in agregadas], dtype=np.float64)])
                combined_labels = np.vstack([combined_labels, np.asarray([e[2] for e in agregadas], dtype=combined_labels.dtype)])

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
        f0_indices = current_fronts[0]
        f0_hv = hypervolume_f1(objectives[f0_indices])
        history_objectives.append(objectives.copy())
        history_labels.append(labels_pop.copy())
        history_hv_f1.append(f0_hv)

        print(
            f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  "
            f"población = {len(population)}  |  F0 = {len(f0_indices)}  |  "
            f"HV F0 = {f0_hv:.6f}"
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
# MODULO MODULE-PARTICIPATION SEARCH (MPS)
# 1) ConstruirMódulos: agrupamiento jerárquico (average) sobre D_BI = 1 − Wang,
#    cortado en L módulos (una sola vez por corrida; Z reutilizable).
# 2) ParticipaciónRelativa: π_l = cnt_l / (|Mód_l|/n · |F0| · K); los módulos
#    con π < 1 (sub-representados en F0) forman el Pool.
# 3) Para cada solución C de F0: z_k al azar, vecindad = fracción q de
#    (Pool \ C) más cercana a z_k según D_BI, con recorte por el presupuesto B.
# 4) Cada candidata reemplaza a z_k, la partición se reconstruye GLOBALMENTE
#    con D_GE y se acepta si mejora AL MENOS UN objetivo respecto a C. El
#    operador retorna solo el frente F0 del conjunto aceptado.
# Si el algoritmo está estancado (MPS no aporta soluciones nuevas a Rv), la
# vecindad pasa a ser una muestra aleatoria de genes fuera de C y de N(z_k),
# del mismo tamaño que N(z_k) (azar → más variabilidad de genes).
#######################################################

_Z_CACHE: dict = {}      # id(bi_matrix) → (weakref a bi_matrix, enlace, Z): el dendrograma solo depende de D_BI.

def build_modules(
    bi_matrix: np.ndarray,
    L: int,
    enlace: str = "average",
    Z: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    ConstruirMódulos(D_BI, L, enlace): agrupa jerárquicamente los genes con
    D_BI (= 1 − Wang) y corta el dendrograma en `L` módulos.

    Z : dendrograma precalculado. Si no se entrega, se busca en un caché por
        matriz (así las corridas/trials sobre el mismo dataset en un mismo
        proceso no repiten el agrupamiento, que es la parte costosa).

    Retorna
    -------
    mod     : np.ndarray (n,) — módulo (1..L) de cada gen.
    modulos : list[np.ndarray] — modulos[l-1] = índices de los genes del módulo l.
    Z       : dendrograma (para reutilizar).

    Nota: con empates de distancia el corte puede devolver menos de L módulos.
    """
    if Z is None:
        hit = _Z_CACHE.get(id(bi_matrix))
        if hit is not None and hit[0]() is bi_matrix and hit[1] == enlace:
            Z = hit[2]
        else:
            Z = linkage(squareform(bi_matrix, checks=False), method=enlace)
            key = id(bi_matrix)
            _Z_CACHE[key] = (weakref.ref(bi_matrix, lambda _r, k=key: _Z_CACHE.pop(k, None)), enlace, Z)
    mod = fcluster(Z, t=L, criterion="maxclust")
    modulos = [np.flatnonzero(mod == l) for l in range(1, int(mod.max()) + 1)]
    return mod, modulos, Z

def relative_participation(
    front_solutions: np.ndarray,
    mod: np.ndarray,
    modulos: list[np.ndarray],
) -> np.ndarray:
    """
    ParticipaciónRelativa(F0, mod, {Mód_l}, K): π_l = cnt_l / esperado_l, con
    cnt_l = medoides de F0 dentro del módulo l y
    esperado_l = (|Mód_l| / n) · |F0| · K.   π_l < 1 ⇒ módulo sub-representado.
    """
    front_solutions = np.asarray(front_solutions)
    n_sol, K = front_solutions.shape
    cnt = np.bincount(mod[front_solutions.ravel()], minlength=len(modulos) + 1)[1:]
    tam = np.array([len(m) for m in modulos], dtype=np.float64)
    esperado = tam / len(mod) * n_sol * K
    return cnt / esperado

def module_participation_search(
    front_solutions: np.ndarray,
    front_objectives: np.ndarray,
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    mod: np.ndarray,
    modulos: list[np.ndarray],
    q: float = 0.10,
    max_obj_calls: int | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    info: dict | None = None,
    azar: bool = False,
) -> list[dict]:
    """
    Module-Participation Search (MPS) sobre el frente F0 `front_solutions`.

    Parámetros
    ----------
    front_solutions  : np.ndarray (m, K) — medoides de cada solución de F0.
    front_objectives : np.ndarray (m, 2) — (XB_GE_0, XB_BI_0) de cada solución.
    ge_matrix, bi_matrix : D_GE y D_BI (n×n, ya normalizadas).
    mod, modulos     : salida de `build_modules`.
    q                : fracción de genes de Cand más cercanos a z_k según D_BI.
    max_obj_calls    : tope B sobre solution_encoding.OBJ_FUNCTION_CALLS
        (cuota GLOBAL). Si es None no hay tope. Si la vecindad excede lo que
        resta, se recorta conservando los más cercanos a z_k.
    rng              : np.random.Generator | None — elección de z_k.
    info             : dict | None — si se entrega, se rellena con diagnósticos
        (pi, modulos_sub, pool_size, evaluadas, aceptadas, unicas).
    azar             : bool — si True (el algoritmo está estancado), la vecindad de
        cada z_k NO usa Pool ni cercanía: es una muestra aleatoria sin reposición de
        genes de TODO el dataset fuera de C y fuera de N(z_k). Su tamaño es
        |N(z_k)| (mismo gasto de cuota que sin estancamiento). La aceptación y el
        filtro final no cambian.

    Retorna
    -------
    list[dict] con llaves "solution", "labels", "xb_ge", "xb_bi": el frente F0 del
    conjunto de soluciones aceptadas (sin duplicados; puede quedar vacío).
    """
    front_solutions = np.asarray(front_solutions)
    front_objectives = np.asarray(front_objectives, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    K = front_solutions.shape[1]

    # ── Participación relativa (se recalcula en cada invocación) y Pool ──
    pi = relative_participation(front_solutions, mod, modulos)
    modulos_sub = np.flatnonzero(pi < 1) + 1                     # Módulos sub-representados (1-based).
    pool_mask = np.isin(mod, modulos_sub)

    aceptadas: dict[frozenset, dict] = {}
    n_evaluadas = 0
    n_aceptadas = 0

    for C, (xb_ge_0, xb_bi_0) in zip(front_solutions, front_objectives):
        k = int(rng.integers(1, K + 1))                          # z_k al azar (se consume rng siempre, para reproducibilidad).
        zk = int(C[k - 1])

        cand_mask = pool_mask.copy()
        cand_mask[C] = False                                     # excluir: TODOS los medoides de C.
        cand = np.flatnonzero(cand_mask)
        if len(cand) == 0:
            continue

        # N(z_k): los ⌈q·|Cand|⌉ genes de Cand con menor D_BI[z_k, ·] (en orden de cercanía).
        tam_vec = int(np.ceil(q * len(cand)))
        orden = np.argsort(bi_matrix[zk, cand], kind="stable")[:tam_vec]
        N_zk = cand[orden]

        if azar:
            # Estancamiento: mismo tamaño que N(z_k), pero genes al azar de todo el
            # dataset que NO pertenecen a N(z_k) ni a C.
            elegibles_mask = np.ones(len(mod), dtype=bool)
            elegibles_mask[C] = False
            elegibles_mask[N_zk] = False
            elegibles = np.flatnonzero(elegibles_mask)
            N_zk = rng.choice(elegibles, size=min(len(N_zk), len(elegibles)), replace=False)

        # ── Presupuesto global de evaluaciones ──
        if max_obj_calls is not None:
            restante = max_obj_calls - solution_encoding.OBJ_FUNCTION_CALLS
            if restante <= 0:
                break
            if len(N_zk) > restante:
                N_zk = N_zk[:restante]                           # recorte por cercanía (o al azar si `azar`, ya barajado).

        if verbose:
            print(f"[MPS] k={k}  z_k={zk}  |Cand|={len(cand)}  |N(z_k)|={len(N_zk)}" + ("  (azar)" if azar else ""))

        # ── Evaluación batched (reconstrucción GLOBAL de la partición con D_GE) ──
        C_primes = np.tile(C, (len(N_zk), 1))
        C_primes[:, k - 1] = N_zk.astype(C.dtype)
        labels_primes = build_clusters_population(ge_matrix, C_primes)
        xb_primes = xie_beni_population(ge_matrix, bi_matrix, C_primes, labels_primes)
        n_evaluadas += len(N_zk)

        # ── Aceptación: mejora en al menos un objetivo ──
        acepta = (xb_primes[:, 0] < xb_ge_0) | (xb_primes[:, 1] < xb_bi_0)
        n_aceptadas += int(acepta.sum())
        for i in np.flatnonzero(acepta):
            clave = frozenset(C_primes[i].tolist())
            if clave not in aceptadas:                           # La misma C' puede surgir de dos soluciones.
                aceptadas[clave] = {
                    "solution": C_primes[i].copy(),
                    "labels": labels_primes[i],
                    "xb_ge": float(xb_primes[i, 0]),
                    "xb_bi": float(xb_primes[i, 1]),
                }

    if info is not None:
        info.update(pi=pi, modulos_sub=modulos_sub, pool_size=int(pool_mask.sum()),
                    evaluadas=n_evaluadas, aceptadas=n_aceptadas, unicas=len(aceptadas))

    # ── Filtro final: solo el frente F0 del conjunto aceptado ──
    if not aceptadas:
        return []
    lista = list(aceptadas.values())
    objs = np.array([[d["xb_ge"], d["xb_bi"]] for d in lista], dtype=np.float64)
    return [lista[i] for i in non_dominated_sort(objs)[0]]


#######################################################
# run_nsga2_MPS: NSGA-II propio que agrega Module-Participation Search
# (MPS) como estrategia de búsqueda local
#######################################################

def run_nsga2_MPS(
    ge_matrix: np.ndarray,
    bi_matrix: np.ndarray,
    n: int,
    k: int,
    pop_size: int,
    max_obj_function_calls: int,
    L: int = 20,
    q: float = 0.10,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    seed: int | None = None,
    Z: np.ndarray | None = None,
    enlace: str = "average",
    gens_estancamiento: int = 1,
) -> dict:
    """
    NSGA-II con MPS: tras construir Rv = Pv ∪ Qv se extrae su frente no
    dominado F0 y se invoca UNA vez MPS sobre todo F0 (una aplicación por
    solución, cada una sobre un medoide al azar). MPS aplica su propio filtro
    (mejora en ≥1 objetivo respecto a la solución de origen + frente F0 del
    conjunto aceptado); las soluciones retornadas que no estaban en Rv se
    agregan a Rv antes de la selección de sobrevivientes. Lo demás (torneo,
    crossover, mutación, truncamiento por rank+crowding) es IDÉNTICO a run_nsga2.

    L, q    : número de módulos y fracción de vecindad de MPS.
    Z       : dendrograma de D_BI precalculado (opcional). Los módulos no
              dependen de la población: se construyen UNA sola vez por corrida
              y el dendrograma se cachea por matriz (ver `build_modules`).

    Estancamiento: la generación en que MPS no aporta soluciones nuevas a Rv
    cuenta como la 1ª; al llegar a `gens_estancamiento` la vecindad de cada z_k
    pasa a ser una muestra aleatoria (mismo tamaño, genes fuera de C y de N(z_k)),
    hasta que MPS vuelva a aportar soluciones nuevas.

    Criterio de paro EXACTO: MPS recibe la cuota global como presupuesto B.
    """
    mod, modulos, _ = build_modules(bi_matrix, L=L, enlace=enlace, Z=Z)

    rng = np.random.default_rng(seed)

    population = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
    population, objectives, labels_pop = evaluate_population(
        population, ge_matrix, bi_matrix, max_obj_function_calls=max_obj_function_calls,
    )

    initial_fronts = non_dominated_sort(objectives) if len(objectives) else [[]]
    initial_f0 = initial_fronts[0]

    history_objectives = [objectives.copy()]
    history_labels = [labels_pop.copy()]
    history_hv_f1 = [hypervolume_f1(objectives[initial_f0])] if len(objectives) else []
    history_mps = []                                         # Por generación: (evaluadas, aceptadas, nuevas agregadas a Rv, en modo azar).

    gen = 0
    gens_sin_cambio = 0                                      # Generaciones seguidas en estancamiento (la que lo suscita cuenta como 1).
    estancado = False
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

        # ── Operador MPS: ÚNICO agregado respecto a run_nsga2 ────────────────
        if solution_encoding.OBJ_FUNCTION_CALLS < max_obj_function_calls:
            f0_idx = non_dominated_sort(combined_objectives)[0]
            rv_keys = {frozenset(sol.tolist()) for sol in combined_population}

            info = {}
            salida = module_participation_search(
                combined_population[f0_idx], combined_objectives[f0_idx],
                ge_matrix, bi_matrix, mod, modulos,
                q=q, max_obj_calls=max_obj_function_calls, rng=rng, info=info,
                azar=estancado,                              # Estancado → muestra aleatoria (fuera de C y de N(z_k)).
            )

            # Solo soluciones REALMENTE nuevas (que no estaban en Rv).
            agregadas = [d for d in salida if frozenset(d["solution"].tolist()) not in rv_keys]
            history_mps.append((info["evaluadas"], info["aceptadas"], len(agregadas), estancado))
            # Estancamiento: MPS no varió Rv (la generación en que ocurre cuenta como la 1ª).
            gens_sin_cambio = 0 if agregadas else gens_sin_cambio + 1
            estancado = gens_sin_cambio >= gens_estancamiento

            if agregadas:
                # Objetivos y labels YA calculados dentro del operador (no
                # re-evaluar: duplicaría el gasto de cuota).
                combined_population = np.vstack([combined_population, np.asarray([d["solution"] for d in agregadas], dtype=combined_population.dtype)])
                combined_objectives = np.vstack([combined_objectives, np.asarray([[d["xb_ge"], d["xb_bi"]] for d in agregadas], dtype=np.float64)])
                combined_labels = np.vstack([combined_labels, np.asarray([d["labels"] for d in agregadas], dtype=combined_labels.dtype)])

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
        f0_indices = current_fronts[0]
        f0_hv = hypervolume_f1(objectives[f0_indices])
        history_objectives.append(objectives.copy())
        history_labels.append(labels_pop.copy())
        history_hv_f1.append(f0_hv)

        print(
            f"[gen {gen:>3}] llamadas = {solution_encoding.OBJ_FUNCTION_CALLS:>6}/{max_obj_function_calls}  |  "
            f"población = {len(population)}  |  F0 = {len(f0_indices)}  |  "
            f"HV F0 = {f0_hv:.6f}  |  MPS: eval={history_mps[-1][0] if history_mps else 0}, nuevas={history_mps[-1][2] if history_mps else 0}"
            + ("  (azar)" if history_mps and history_mps[-1][3] else "")
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
        "history_mps": history_mps,
        "generations_run": gen,
    }
