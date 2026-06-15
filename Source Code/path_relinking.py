"""
path_relinking.py
------------------
Implementación de Multi-Objective Path-Relinking (MOPR) para un par de
soluciones, según Parraga-Alava et al. (2018).

Algoritmo (resumen del paper)
------------------------------
Dadas dos soluciones C1 (start) y C2 (guide), ambas representadas como
arreglos de k medoides:

    1. MC1 = medoides en C1, MC2 = medoides en C2.
    2. MC1 - MC2: medoides en C1 que NO están en C2 (a remover).
       MC2 - MC1: medoides en C2 que NO están en C1 (a agregar).
    3. tr0(C1,C2) = C1.
    4. Para obtener tr_{i+1}, se remueve un medoide z_k ∈ (MC1-MC2) restante
       y se agrega un medoide z_l ∈ (MC2-MC1) restante, generando TODAS las
       combinaciones posibles de swap como candidatas.
    5. Las candidatas se evalúan con XBEB y XBBB (objetivos de
       minimización). Se aplica ordenamiento por dominancia de Pareto
       (non-dominated sorting) + crowding distance, y se selecciona la
       candidata mejor rankeada como nuevo tr_{i+1}.
    6. Se repite hasta que no queden medoides por remover (la trayectoria
       converge al conjunto de medoides de C2).

Para un par (C1, C2), se aplica PR(C1,C2) y PR(C2,C1). Las trayectorias se
fusionan junto con C1 y C2 originales, formando un "intermediate pool" (IP).
Se filtran duplicados (mismo conjunto de medoides) y se aplica un nuevo
ordenamiento por dominancia; las soluciones no dominadas (F1) son el
resultado final del procedimiento.

Objetivos utilizados
--------------------
Se usan XBEB y XBBB (Xie-Beni con distancia de expresión y biológica
respectivamente), siguiendo la combinación ganadora reportada en el paper.
Cada objetivo requiere su propia asignación de clusters (build_clusters
con DEB para XBEB, con DBB para XBBB), tal como se describe en la Tabla 1
del artículo.

Nota
----
La evaluación de objetivos (evaluate_objectives) y el ordenamiento por
dominancia de Pareto + crowding distance (non_dominated_sort,
crowding_distance) se importan desde pareto_sorting.py, donde también
están disponibles a nivel de población completa y con visualización de
los frentes de Pareto.
"""

##################################
# Imports
##################################
import numpy as np
import pandas as pd
from pathlib import Path

from solution_definition import (
    load_distance_matrix,
    medoids_from_list,
    build_clusters_population,
    save_clusters,
)
from crossover_mutation import _validate_chromosome
from pareto_sorting import (
    evaluate_objectives,
    non_dominated_sort,
    crowding_distance,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SELECCIÓN DEL MEJOR MOVIMIENTO (Pareto + crowding distance)
# ══════════════════════════════════════════════════════════════════════════════

def _select_best(objectives: list[tuple[float, float]]) -> int:
    """
    Selecciona el mejor índice de un conjunto de candidatas:
    primero por frente de Pareto (F1), y en caso de empate, por la
    mayor crowding distance.

    Parámetros
    ----------
    objectives : list[tuple[float, float]] — (XBEB, XBBB) de cada candidata.

    Retorna
    -------
    int — índice de la candidata seleccionada.
    """
    fronts = non_dominated_sort(objectives)
    f1 = fronts[0]

    if len(f1) == 1:
        return f1[0]

    dist = crowding_distance(objectives, f1)
    return max(f1, key=lambda i: dist[i])


# ══════════════════════════════════════════════════════════════════════════════
# 2. PATH-RELINKING (UNA DIRECCIÓN)
# ══════════════════════════════════════════════════════════════════════════════

def path_relinking(
    start: np.ndarray,
    guide: np.ndarray,
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    verbose: bool = False,
) -> list[np.ndarray]:
    """
    Construye la trayectoria PR(start, guide): transforma gradualmente
    `start` en `guide` intercambiando un medoide a la vez, seleccionando
    en cada paso el mejor movimiento mediante dominancia de Pareto +
    crowding distance sobre (XBEB, XBBB).

    Parámetros
    ----------
    start      : np.ndarray — solución inicial (arreglo de k medoides, 0-based).
    guide      : np.ndarray — solución guía (arreglo de k medoides, 0-based).
    deb_matrix : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    dbb_matrix : np.ndarray — matriz (n×n) de distancia biológica (DBB).
    verbose    : bool       — si True, imprime cada paso de la trayectoria.

    Retorna
    -------
    list[np.ndarray] — trayectoria TR(start, guide): lista de arreglos de
    medoides, comenzando en `start` y terminando en una solución cuyo
    conjunto de medoides coincide con el de `guide`.

    Ejemplo
    -------
    >>> trajectory = path_relinking(C1, C2, deb_matrix, dbb_matrix)
    >>> print(len(trajectory))  # número de pasos + 1
    """
    n = deb_matrix.shape[0]
    _validate_chromosome(start, n, "start")
    _validate_chromosome(guide, n, "guide")
    if start.shape != guide.shape:
        raise ValueError(
            f"'start' y 'guide' deben tener la misma longitud: "
            f"{start.shape} vs {guide.shape}."
        )

    set_start = set(start.tolist())
    set_guide = set(guide.tolist())

    # Medoides a remover de 'start' y a agregar desde 'guide'.
    to_remove = sorted(set_start - set_guide)
    to_add = sorted(set_guide - set_start)
    assert len(to_remove) == len(to_add), (
        "El número de medoides a remover y agregar debe coincidir "
        "(ambos cromosomas deben tener la misma longitud k)."
    )

    current = start.copy()
    trajectory = [current.copy()]

    if verbose:
        print(f"[PR] start={start.tolist()}  guide={guide.tolist()}")
        print(f"[PR] to_remove={to_remove}  to_add={to_add}")

    while to_remove:
        candidates = []
        moves = []

        for zk in to_remove:
            pos = int(np.where(current == zk)[0][0])
            for zl in to_add:
                candidate = current.copy()
                candidate[pos] = zl
                candidates.append(candidate)
                moves.append((zk, zl))

        objectives = [
            evaluate_objectives(cand, deb_matrix, dbb_matrix) for cand in candidates
        ]

        best_idx = _select_best(objectives)
        current = candidates[best_idx]
        zk_chosen, zl_chosen = moves[best_idx]

        to_remove.remove(zk_chosen)
        to_add.remove(zl_chosen)
        trajectory.append(current.copy())

        if verbose:
            xbeb, xbbb = objectives[best_idx]
            print(
                f"[PR] swap z_{zk_chosen} → z_{zl_chosen}  "
                f"→ {current.tolist()}  (XBEB={xbeb:.6f}, XBBB={xbbb:.6f})"
            )

    return trajectory


# ══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-OBJECTIVE PATH-RELINKING (AMBAS DIRECCIONES + FUSIÓN)
# ══════════════════════════════════════════════════════════════════════════════

def multi_objective_path_relinking(
    C1: np.ndarray,
    C2: np.ndarray,
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    verbose: bool = False,
) -> dict:
    """
    Aplica MOPR completo sobre un par de soluciones (C1, C2):

        1. PR(C1, C2) y PR(C2, C1).
        2. Fusiona ambas trayectorias junto con C1 y C2 originales,
           formando el intermediate pool (IP).
        3. Elimina soluciones duplicadas (mismo conjunto de medoides).
        4. Evalúa XBEB y XBBB para todo el pool.
        5. Aplica non-dominated sorting; retorna F1 como resultado.

    Parámetros
    ----------
    C1         : np.ndarray — primera solución (arreglo de k medoides, 0-based).
    C2         : np.ndarray — segunda solución (arreglo de k medoides, 0-based).
    deb_matrix : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    dbb_matrix : np.ndarray — matriz (n×n) de distancia biológica (DBB).
    verbose    : bool       — si True, imprime el progreso de ambas trayectorias.

    Retorna
    -------
    dict con claves:
        "pool"        : np.ndarray — todas las soluciones únicas del IP, shape (m, k).
        "pool_obj"    : list[tuple[float, float]] — (XBEB, XBBB) de cada solución del pool.
        "f1"          : np.ndarray — soluciones no dominadas, shape (|F1|, k).
        "f1_obj"      : list[tuple[float, float]] — (XBEB, XBBB) de cada solución de F1.
        "trajectory_C1_to_C2" : list[np.ndarray] — trayectoria PR(C1, C2).
        "trajectory_C2_to_C1" : list[np.ndarray] — trayectoria PR(C2, C1).

    Ejemplo
    -------
    >>> result = multi_objective_path_relinking(C1, C2, deb_matrix, dbb_matrix)
    >>> print(result["f1"])
    >>> print(result["f1_obj"])
    """
    if verbose:
        print("\n[MOPR] Trayectoria PR(C1, C2)")
    traj_c1_c2 = path_relinking(C1, C2, deb_matrix, dbb_matrix, verbose=verbose)

    if verbose:
        print("\n[MOPR] Trayectoria PR(C2, C1)")
    traj_c2_c1 = path_relinking(C2, C1, deb_matrix, dbb_matrix, verbose=verbose)

    # Fusionar y eliminar duplicados por conjunto de medoides.
    all_solutions = traj_c1_c2 + traj_c2_c1
    unique_pool: dict[frozenset, np.ndarray] = {}
    for sol in all_solutions:
        key = frozenset(sol.tolist())
        if key not in unique_pool:
            unique_pool[key] = sol

    pool = np.stack(list(unique_pool.values()))

    # Evaluar objetivos para todo el pool.
    pool_obj = [evaluate_objectives(ind, deb_matrix, dbb_matrix) for ind in pool]

    # Non-dominated sorting → F1.
    fronts = non_dominated_sort(pool_obj)
    f1_indices = fronts[0]

    f1 = pool[f1_indices]
    f1_obj = [pool_obj[i] for i in f1_indices]

    if verbose:
        print(f"\n[MOPR] Pool fusionado: {pool.shape[0]} soluciones únicas.")
        print(f"[MOPR] Frente no dominado (F1): {len(f1_indices)} soluciones.")
        for sol, (xbeb, xbbb) in zip(f1, f1_obj):
            print(f"  {sol.tolist()}  → XBEB={xbeb:.6f}, XBBB={xbbb:.6f}")

    return {
        "pool": pool,
        "pool_obj": pool_obj,
        "f1": f1,
        "f1_obj": f1_obj,
        "trajectory_C1_to_C2": traj_c1_c2,
        "trajectory_C2_to_C1": traj_c2_c1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def save_path_relinking_results(
    result: dict,
    gene_names: list[str],
    deb_matrix: np.ndarray,
    output_dir: str,
    prefix: str = "pr",
) -> dict:
    """
    Guarda los resultados de multi_objective_path_relinking() en dos archivos:

        - "{prefix}_f1_objectives.csv": tabla con los medoides (nombres de
          gen) y los valores XBEB / XBBB de cada solución del frente F1.
        - "{prefix}_f1_clusters.csv": asignación de clusters (formato
          solucionN × genes, 1-based) de cada solución de F1, calculada
          con DEB, en el mismo formato usado por medoids_clustering.py.

    Parámetros
    ----------
    result     : dict        — salida de multi_objective_path_relinking().
    gene_names : list[str]   — nombres de genes, extraídos de load_distance_matrix().
    deb_matrix : np.ndarray   — matriz (n×n) de distancia de expresión (DEB),
                                usada para calcular los clusters de F1.
    output_dir : str         — carpeta destino (se crea si no existe).
    prefix     : str         — prefijo para los nombres de archivo.

    Retorna
    -------
    dict con claves:
        "objectives_path" : str — ruta del CSV de objetivos.
        "clusters_path"   : str — ruta del CSV de asignación de clusters.

    Ejemplo
    -------
    >>> paths = save_path_relinking_results(result, gene_names, deb, output_dir="resultados")
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    f1 = result["f1"]
    f1_obj = result["f1_obj"]
    k = f1.shape[1]

    # ── Tabla de objetivos ──────────────────────────────────────────────────
    rows = []
    for i, (medoids, (xbeb, xbbb)) in enumerate(zip(f1, f1_obj)):
        row = {"solution_id": f"f1_{i + 1}"}
        for m in range(k):
            row[f"medoid_{m + 1}"] = gene_names[medoids[m]]
        row["XBEB"] = xbeb
        row["XBBB"] = xbbb
        rows.append(row)

    df_obj = pd.DataFrame(rows)
    obj_path = out_path / f"{prefix}_f1_objectives.csv"
    df_obj.to_csv(obj_path, index=False)
    print(f"[save] Objetivos de F1 guardados en: {obj_path}")

    # ── Asignación de clusters (EB) para cada solución de F1 ────────────────
    all_labels = build_clusters_population(deb_matrix, f1)
    clusters_path_str = save_clusters(
        all_labels=all_labels,
        gene_names=gene_names,
        output_dir=output_dir,
        filename=f"{prefix}_f1_clusters.csv",
    )

    return {
        "objectives_path": str(obj_path),
        "clusters_path": clusters_path_str,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def run_path_relinking(
    deb_path: str,
    dbb_path: str,
    C1_indices: list[int],
    C2_indices: list[int],
    output_dir: str,
    prefix: str = "pr",
    verbose: bool = True,
) -> dict:
    """
    Pipeline completo: carga matrices → aplica MOPR sobre (C1, C2) → guarda resultados.

    Parámetros
    ----------
    deb_path   : str        — ruta a la matriz DEB (.csv).
    dbb_path   : str        — ruta a la matriz DBB (.csv).
    C1_indices : list[int]  — medoides de la primera solución (índices 0-based).
    C2_indices : list[int]  — medoides de la segunda solución (índices 0-based).
    output_dir : str        — carpeta de salida.
    prefix     : str        — prefijo para los archivos de salida.
    verbose    : bool       — si True, imprime el progreso del MOPR.

    Retorna
    -------
    dict con claves:
        "result"          : dict — salida de multi_objective_path_relinking().
        "gene_names"      : list[str].
        "objectives_path" : str — ruta del CSV de objetivos de F1.
        "clusters_path"   : str — ruta del CSV de clusters de F1.

    Ejemplo
    -------
    >>> output = run_path_relinking(
    ...     deb_path="data/DEB.csv",
    ...     dbb_path="data/DBB.csv",
    ...     C1_indices=[10, 25, 3, 47],
    ...     C2_indices=[5, 19, 33, 8],
    ...     output_dir="resultados/pr1",
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
    n = deb.shape[0]

    C1 = medoids_from_list(C1_indices, n=n)
    C2 = medoids_from_list(C2_indices, n=n)

    print(f"[MOPR] C1 = {C1.tolist()} → {[gene_names[i] for i in C1]}")
    print(f"[MOPR] C2 = {C2.tolist()} → {[gene_names[i] for i in C2]}")

    result = multi_objective_path_relinking(C1, C2, deb, dbb, verbose=verbose)

    paths = save_path_relinking_results(
        result=result,
        gene_names=gene_names,
        deb_matrix=deb,
        output_dir=output_dir,
        prefix=prefix,
    )

    return {
        "result": result,
        "gene_names": gene_names,
        **paths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    deb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBE.csv"
    dbb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBI.csv"
    output_dir = r"C:\Users\benja\Desktop\workspace\Thesis\Results"

    # Dos soluciones de ejemplo (medoides 0-based), deben tener la misma
    # longitud k y compartir AL MENOS un medoide en distinto para que la
    # trayectoria tenga al menos un paso.
    C1_indices = [10, 25, 3, 47]
    C2_indices = [5, 19, 33, 8]

    output = run_path_relinking(
        deb_path=deb_path,
        dbb_path=dbb_path,
        C1_indices=C1_indices,
        C2_indices=C2_indices,
        output_dir=output_dir,
        prefix="pr_example",
    )

    print("\n── Frente F1 resultante ──")
    for sol, (xbeb, xbbb) in zip(output["result"]["f1"], output["result"]["f1_obj"]):
        genes = [output["gene_names"][i] for i in sol]
        print(f"  {genes}  → XBEB={xbeb:.6f}, XBBB={xbbb:.6f}")