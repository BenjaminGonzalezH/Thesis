"""
pareto_local_search.py
-----------------------
Implementación de Pareto Local Search (PLS) como estrategia de
diversificación de MOC-GaPBK (Parraga-Alava et al., 2018).

Algoritmo (resumen del paper, Fig. 3)
---------------------------------------
PLS recibe una población de soluciones no dominadas A0 (todas marcadas
como "no exploradas") y la duplica en una población de trabajo A.

Mientras A0 no esté vacía, se repite un proceso de tres pasos:

    1. Selección: se elige aleatoriamente una solución C de A0.

    2. Exploración de vecindad: se elige al azar un medoide z_k de C
       (una posición del cromosoma) y se genera el vecindario N(C),
       reemplazando z_k por cada medoide z_l que aún no esté presente
       en C. Esto produce |N(C)| = n - k soluciones vecinas, cada una
       con un único medoide distinto de C.

    3. Criterio de aceptación (dominancia): para cada C' ∈ N(C), si C
       NO domina a C' (C ⊀ C'):
           - C' se agrega a A marcada como no explorada.
           - Se eliminan de A todas las soluciones dominadas por C'
             (lo que puede incluir a la propia C).

    Finalmente, C se marca como explorada. A0 se actualiza con las
    soluciones aún no exploradas de A.

PLS termina cuando A0 queda vacía (todas las soluciones de A han sido
exploradas). La población A resultante corresponde a la nueva Población R
(Algoritmo 1, líneas 13-14 del paper), sobre la cual se vuelve a aplicar
NON-DOMINATEDSORTING-CROWDINGDISTANCE.

Objetivos utilizados
--------------------
Como en path_relinking.py, se usan XBEB y XBBB (combinación ganadora del
paper), cada uno con su propia asignación de clusters (DEB para XBEB, DBB
para XBBB). evaluate_objectives, _dominates y
non_dominated_sorting_crowding_distance se importan desde pareto_sorting.py.
"""

##################################
# Imports
##################################
import numpy as np
import pandas as pd
from pathlib import Path

from solution_definition import (
    load_distance_matrix,
    build_clusters_population,
    save_clusters,
)
from pareto_sorting import (
    evaluate_objectives,
    evaluate_population_objectives,
    _dominates,
    non_dominated_sorting_crowding_distance,
    plot_pareto_fronts,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXPLORACIÓN DE VECINDAD
# ══════════════════════════════════════════════════════════════════════════════

def generate_neighborhood(
    C: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int, int]:
    """
    Genera el vecindario N(C) de una solución, reemplazando un medoide
    elegido al azar por cada elemento del dataset aún no presente en C.

    Parámetros
    ----------
    C   : np.ndarray — solución actual, arreglo de k medoides (0-based).
    n   : int        — número total de elementos del dataset.
    rng : np.random.Generator — generador aleatorio.

    Retorna
    -------
    tuple con:
        neighbors : list[np.ndarray] — |N(C)| = n - k vecinos, cada uno
                    con un único medoide distinto respecto a C.
        pos       : int — posición del cromosoma que fue modificada.
        zk        : int — medoide original en esa posición (el que se
                    intenta reemplazar).

    Ejemplo
    -------
    >>> rng = np.random.default_rng(0)
    >>> neighbors, pos, zk = generate_neighborhood(C, n=100, rng=rng)
    >>> print(len(neighbors))  # n - k
    """
    k = len(C)
    pos = int(rng.integers(0, k))
    zk = int(C[pos])

    used = set(C.tolist())
    candidates_z = [z for z in range(n) if z not in used]

    neighbors = []
    for zl in candidates_z:
        neighbor = C.copy()
        neighbor[pos] = zl
        neighbors.append(neighbor)

    return neighbors, pos, zk


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARETO LOCAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def pareto_local_search(
    initial_population: np.ndarray,
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    max_iterations: int | None = 200,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Aplica Pareto Local Search (PLS) sobre una población inicial de
    soluciones no dominadas.

    Mantiene un pool de soluciones (indexado por su conjunto de medoides
    para evitar duplicados) y explora vecindarios hasta que no queden
    soluciones sin explorar, o hasta alcanzar `max_iterations`.

    Parámetros
    ----------
    initial_population : np.ndarray — población inicial A0, shape (m, k).
                          Idealmente el frente F1 de una etapa previa
                          (ej. salida de multi_objective_path_relinking()).
    deb_matrix          : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    dbb_matrix          : np.ndarray — matriz (n×n) de distancia biológica (DBB).
    max_iterations      : int | None — número máximo de selecciones (paso 1).
                          Si es None, se ejecuta hasta que A0 esté vacía
                          (puede ser costoso para n grande). Default: 200.
    seed                : int — semilla para reproducibilidad (opcional).
    verbose             : bool — si True, imprime el progreso de cada iteración.

    Retorna
    -------
    tuple con:
        population : np.ndarray — población final A, shape (m', k).
        objectives : list[tuple[float, float]] — (XBEB, XBBB) de cada
                     solución de `population`, en el mismo orden.

    Ejemplo
    -------
    >>> final_pop, final_obj = pareto_local_search(
    ...     initial_population=f1_population,
    ...     deb_matrix=deb, dbb_matrix=dbb,
    ...     max_iterations=200, seed=42,
    ... )
    """
    rng = np.random.default_rng(seed)
    n = deb_matrix.shape[0]

    # Pool: frozenset(medoides) → {"solution", "objectives", "explored"}
    pool: dict[frozenset, dict] = {}
    for sol in initial_population:
        key = frozenset(sol.tolist())
        if key not in pool:
            obj = evaluate_objectives(sol, deb_matrix, dbb_matrix)
            pool[key] = {"solution": sol.copy(), "objectives": obj, "explored": False}

    if verbose:
        print(f"[PLS] Población inicial A0: {len(pool)} soluciones únicas.")

    iteration = 0
    while True:
        unexplored_keys = [key for key, entry in pool.items() if not entry["explored"]]

        if not unexplored_keys:
            if verbose:
                print(f"[PLS] A0 vacía tras {iteration} iteraciones. Población final |A|={len(pool)}.")
            break

        if max_iterations is not None and iteration >= max_iterations:
            if verbose:
                print(f"[PLS] Límite de iteraciones alcanzado ({max_iterations}). "
                      f"Quedan {len(unexplored_keys)} soluciones sin explorar. |A|={len(pool)}.")
            break

        iteration += 1

        # ── 1. Selección ────────────────────────────────────────────────────
        c_key = unexplored_keys[int(rng.integers(0, len(unexplored_keys)))]
        C = pool[c_key]["solution"]
        C_obj = pool[c_key]["objectives"]

        # ── 2. Exploración de vecindad ──────────────────────────────────────
        neighbors, pos, zk = generate_neighborhood(C, n, rng)

        if verbose:
            print(f"[PLS] it={iteration:>4}  C={C.tolist()}  "
                  f"(XBEB={C_obj[0]:.4f}, XBBB={C_obj[1]:.4f})  "
                  f"reemplazando posición {pos} (z_{zk}), |N(C)|={len(neighbors)}")

        # ── 3. Criterio de aceptación (dominancia) ──────────────────────────
        for neighbor in neighbors:
            neighbor_key = frozenset(neighbor.tolist())
            neighbor_obj = evaluate_objectives(neighbor, deb_matrix, dbb_matrix)

            # Si C NO domina a C' (C ⊀ C'), se acepta C' en el pool.
            if not _dominates(C_obj, neighbor_obj):
                if neighbor_key not in pool:
                    pool[neighbor_key] = {
                        "solution": neighbor,
                        "objectives": neighbor_obj,
                        "explored": False,
                    }

                # Eliminar del pool soluciones dominadas por C' (puede incluir a C).
                dominated_keys = [
                    key for key, entry in pool.items()
                    if key != neighbor_key and _dominates(neighbor_obj, entry["objectives"])
                ]
                for key in dominated_keys:
                    del pool[key]

        # Marcar C como explorada (si sigue en el pool: pudo ser eliminada
        # si quedó dominada por uno de sus propios vecinos).
        if c_key in pool:
            pool[c_key]["explored"] = True

    population = np.stack([entry["solution"] for entry in pool.values()])
    objectives = [entry["objectives"] for entry in pool.values()]

    return population, objectives


# ══════════════════════════════════════════════════════════════════════════════
# 3. GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def save_pls_results(
    population: np.ndarray,
    objectives: list[tuple[float, float]],
    gene_names: list[str],
    deb_matrix: np.ndarray,
    output_dir: str,
    prefix: str = "pls",
    plot: bool = True,
) -> dict:
    """
    Guarda los resultados de pareto_local_search() en disco:

        - "{prefix}_objectives.csv": medoides (nombres de gen) + XBEB/XBBB
          + rank + crowding distance, para cada solución de la población final.
        - "{prefix}_clusters.csv": asignación de clusters (formato
          solucionN × genes, 1-based), calculada con DEB.
        - "{prefix}_pareto_fronts.png": gráfico de frentes de Pareto (opcional).

    Parámetros
    ----------
    population : np.ndarray — población final de PLS, shape (m, k).
    objectives : list[tuple[float, float]] — (XBEB, XBBB) de cada individuo.
    gene_names : list[str]  — nombres de genes, de load_distance_matrix().
    deb_matrix : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    output_dir : str        — carpeta destino (se crea si no existe).
    prefix     : str        — prefijo para los archivos de salida.
    plot       : bool       — si True, genera el gráfico de frentes de Pareto.

    Retorna
    -------
    dict con claves:
        "objectives_path" : str — ruta del CSV de objetivos.
        "clusters_path"   : str — ruta del CSV de clusters.
        "plot_path"       : str | None — ruta del gráfico, si plot=True.
        "ranks"           : np.ndarray — rank de cada individuo.
        "crowding"        : np.ndarray — crowding distance de cada individuo.
        "fronts"          : list[list[int]] — frentes de Pareto de la población final.

    Ejemplo
    -------
    >>> paths = save_pls_results(population, objectives, gene_names, deb, output_dir="resultados")
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    objectives_arr = np.asarray(objectives, dtype=np.float64)
    ranks, crowding, fronts = non_dominated_sorting_crowding_distance(objectives_arr)

    pop_size, k = population.shape
    rows = []
    for i in range(pop_size):
        row = {"individuo": f"solucion{i + 1}"}
        for m in range(k):
            row[f"medoid_{m + 1}"] = gene_names[population[i, m]]
        row["XBEB"] = objectives_arr[i, 0]
        row["XBBB"] = objectives_arr[i, 1]
        row["rank"] = int(ranks[i])
        row["crowding_distance"] = crowding[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    obj_path = out_path / f"{prefix}_objectives.csv"
    df.to_csv(obj_path, index=False)
    print(f"[save] Objetivos guardados en: {obj_path}")

    all_labels = build_clusters_population(deb_matrix, population)
    clusters_path = save_clusters(
        all_labels=all_labels,
        gene_names=gene_names,
        output_dir=output_dir,
        filename=f"{prefix}_clusters.csv",
    )

    plot_path = None
    if plot:
        plot_path = plot_pareto_fronts(
            objectives_arr, fronts=fronts,
            output_dir=output_dir, filename=f"{prefix}_pareto_fronts.png",
            title="Frentes de Pareto tras Pareto Local Search",
        )

    return {
        "objectives_path": str(obj_path),
        "clusters_path": clusters_path,
        "plot_path": plot_path,
        "ranks": ranks,
        "crowding": crowding,
        "fronts": fronts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def run_pareto_local_search(
    deb_path: str,
    dbb_path: str,
    initial_population: np.ndarray,
    output_dir: str,
    max_iterations: int | None = 200,
    seed: int | None = None,
    prefix: str = "pls",
    verbose: bool = True,
) -> dict:
    """
    Pipeline completo: carga matrices → aplica PLS sobre una población
    inicial → ordena el resultado → guarda objetivos, clusters y gráfico.

    Parámetros
    ----------
    deb_path           : str        — ruta a la matriz DEB (.csv).
    dbb_path           : str        — ruta a la matriz DBB (.csv).
    initial_population : np.ndarray — población inicial A0, shape (m, k).
    output_dir         : str        — carpeta de salida.
    max_iterations     : int | None — ver pareto_local_search().
    seed               : int         — semilla para reproducibilidad.
    prefix             : str         — prefijo para los archivos de salida.
    verbose            : bool        — si True, imprime el progreso de PLS.

    Retorna
    -------
    dict con claves:
        "population"      : np.ndarray — población final de PLS.
        "objectives"       : list[tuple[float, float]].
        "gene_names"       : list[str].
        "objectives_path"  : str.
        "clusters_path"    : str.
        "plot_path"        : str | None.
        "ranks", "crowding", "fronts" : salidas de non_dominated_sorting_crowding_distance.

    Ejemplo
    -------
    >>> output = run_pareto_local_search(
    ...     deb_path="data/DEB.csv", dbb_path="data/DBB.csv",
    ...     initial_population=f1_population,
    ...     output_dir="resultados/pls1", seed=42,
    ... )
    """
    deb, gene_names_deb = load_distance_matrix(deb_path)
    dbb, gene_names_dbb = load_distance_matrix(dbb_path)

    if deb.shape != dbb.shape:
        raise ValueError(
            f"Las matrices DEB {deb.shape} y DBB {dbb.shape} deben tener el mismo tamaño."
        )
    if gene_names_deb != gene_names_dbb:
        raise ValueError(
            "Los nombres de genes en DEB y DBB no coinciden. "
            "Verifica que ambas matrices correspondan al mismo dataset."
        )

    gene_names = gene_names_deb

    if verbose:
        print(f"[PLS] Iniciando con {initial_population.shape[0]} soluciones "
              f"(k={initial_population.shape[1]}, n={deb.shape[0]}).")

    population, objectives = pareto_local_search(
        initial_population=initial_population,
        deb_matrix=deb, dbb_matrix=dbb,
        max_iterations=max_iterations, seed=seed, verbose=verbose,
    )

    if verbose:
        print(f"[PLS] Población final: {population.shape[0]} soluciones.")

    paths = save_pls_results(
        population=population,
        objectives=objectives,
        gene_names=gene_names,
        deb_matrix=deb,
        output_dir=output_dir,
        prefix=prefix,
    )

    return {
        "population": population,
        "objectives": objectives,
        "gene_names": gene_names,
        **paths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from solution_definition import load_distance_matrix as _load, random_medoids_pop

    deb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBE.csv"
    dbb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBI.csv"
    output_dir = r"C:\Users\benja\Desktop\workspace\Thesis\Results"
    k = 4
    seed = 42

    # Población inicial de ejemplo (idealmente sería el F1 de una etapa previa,
    # ej. multi_objective_path_relinking()).
    deb, gene_names = _load(deb_path)
    n = len(gene_names)
    initial_population = random_medoids_pop(n=n, k=k, pop_size=5, seed=seed)

    output = run_pareto_local_search(
        deb_path=deb_path,
        dbb_path=dbb_path,
        initial_population=initial_population,
        output_dir=output_dir,
        max_iterations=200,
        seed=seed,
        prefix="pls_example",
    )

    print("\n── Población final de PLS ──")
    for sol, obj in zip(output["population"], output["objectives"]):
        genes = [output["gene_names"][i] for i in sol]
        print(f"  {genes}  → XBEB={obj[0]:.6f}, XBBB={obj[1]:.6f}")