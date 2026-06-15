"""
MOC_GaPBK.py
-------------
Implementación completa de la metaheurística MOC-GaPBK (Multi-Objective
Clustering Guided by aPriori Biological Knowledge), siguiendo el flujo del
Algoritmo 1 de Parraga-Alava et al. (2018), ensamblada exclusivamente a
partir de los módulos ya construidos:

    - solution_definition.py : lectura de matrices, población de medoides,
                               construcción de clusters, índice Xie-Beni.
    - crossover_mutation.py  : operadores genéticos ((k-1)-point crossover,
                               controller-random mutation) a nivel de población.
    - pareto_sorting.py      : evaluación de objetivos (XBEB, XBBB),
                               non-dominated sorting, crowding distance,
                               visualización de frentes de Pareto.
    - path_relinking.py      : Multi-Objective Path-Relinking (intensificación).
    - pareto_local_search.py : Pareto Local Search (diversificación).

Flujo (Algoritmo 1 del paper)
------------------------------
    1.  Calcular DEB y DBB (ya provistas como .csv → load_distance_matrix).
    2.  Crear población inicial P0 de tamaño N (random_medoids_pop).
    3.  Mientras NO se cumpla el criterio de paro:
        a.  Selección de padres (torneo binario por rank + crowding distance).
        b.  Crossover + mutación → offspring Qv (apply_operators_population).
        c.  Rv = Pv ∪ Qv.
        d.  Rv = NON-DOMINATEDSORTING-CROWDINGDISTANCE(Rv); conservar F1.
        e.  Intensificación: Multi-Objective Path-Relinking sobre pares de F1.
        f.  Rv = NON-DOMINATEDSORTING-CROWDINGDISTANCE(Rv).
        g.  Diversificación: Pareto Local Search sobre F1.
        h.  Rv = NON-DOMINATEDSORTING-CROWDINGDISTANCE(Rv).
        i.  Construir Pv+1 a partir de F1 (completando con población random
            si |F1| < N).
    4.  Retornar las soluciones no dominadas (Pareto Front).
    5.  Guardar la imagen final de las fronteras de Pareto.

Criterio de paro
----------------
A diferencia del paper (que usa número de generaciones), el flujo se controla
mediante un MÁXIMO DE LLAMADAS A LA FUNCIÓN OBJETIVO. Cada evaluación de un
índice Xie-Beni cuenta como 1 llamada; como cada solución se evalúa en sus dos
versiones (XBEB y XBBB), evaluar una solución suma 2 al contador. El algoritmo
verifica el contador antes de cada operación costosa y termina la generación en
curso de forma controlada al alcanzar el límite.
"""

##################################
# Imports
##################################
import numpy as np
import pandas as pd
from pathlib import Path

import pareto_sorting
import path_relinking
import pareto_local_search

from solution_definition import (
    load_distance_matrix,
    random_medoids_pop,
    build_clusters_population,
    save_clusters,
)
from crossover_mutation import apply_operators_population
from pareto_sorting import (
    non_dominated_sorting_crowding_distance,
    plot_pareto_fronts,
)
from path_relinking import multi_objective_path_relinking
from pareto_local_search import pareto_local_search


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTADOR DE LLAMADAS A LA FUNCIÓN OBJETIVO
# ══════════════════════════════════════════════════════════════════════════════

class ObjectiveCounter:
    """
    Contador global de llamadas a la función objetivo (índices Xie-Beni).

    Envuelve la función `evaluate_objectives` de pareto_sorting de modo que
    cada evaluación de una solución incremente el contador. Como cada
    solución se evalúa en sus dos versiones (XBEB y XBBB), cada llamada a
    `evaluate_objectives` suma 2 al total (cada índice cuenta como 1 llamada).

    El parcheo se aplica sobre TODOS los namespaces donde `evaluate_objectives`
    fue importada (pareto_sorting, path_relinking, pareto_local_search), ya que
    cada módulo mantiene su propia referencia tras el `from ... import`.

    Uso
    ---
    >>> counter = ObjectiveCounter(max_calls=10000)
    >>> counter.patch()
    >>> ...  # ejecutar el algoritmo
    >>> counter.restore()
    >>> print(counter.count)
    """

    # Módulos que importaron evaluate_objectives en su namespace.
    _TARGET_MODULES = (pareto_sorting, path_relinking, pareto_local_search)

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.count = 0
        self._original = pareto_sorting.evaluate_objectives
        self._patched = False

    def _wrapper(self, medoids, deb_matrix, dbb_matrix):
        # Cada solución evalúa XBEB y XBBB → 2 llamadas a función objetivo.
        result = self._original(medoids, deb_matrix, dbb_matrix)
        self.count += 2
        return result

    def patch(self) -> None:
        """Reemplaza evaluate_objectives por el wrapper en todos los módulos."""
        if self._patched:
            return
        for mod in self._TARGET_MODULES:
            if hasattr(mod, "evaluate_objectives"):
                setattr(mod, "evaluate_objectives", self._wrapper)
        self._patched = True

    def restore(self) -> None:
        """Restaura la función original en todos los módulos."""
        if not self._patched:
            return
        for mod in self._TARGET_MODULES:
            if hasattr(mod, "evaluate_objectives"):
                setattr(mod, "evaluate_objectives", self._original)
        self._patched = False

    def budget_exhausted(self) -> bool:
        """True si se alcanzó o superó el máximo de llamadas permitido."""
        return self.count >= self.max_calls

    def remaining(self) -> int:
        """Número de llamadas restantes antes de alcanzar el límite."""
        return max(0, self.max_calls - self.count)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SELECCIÓN POR TORNEO BINARIO (rank + crowding distance)
# ══════════════════════════════════════════════════════════════════════════════

def binary_tournament_selection(
    population: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    n_offspring: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Selecciona padres mediante torneo binario, usando ranking de Pareto y,
    en caso de empate, crowding distance (NSGA-II).

    Parámetros
    ----------
    population  : np.ndarray — población actual, shape (pop_size, k).
    ranks       : np.ndarray — rank (nivel de no-dominancia) de cada individuo.
    crowding    : np.ndarray — crowding distance de cada individuo.
    n_offspring : int        — número de padres a seleccionar.
    rng         : np.random.Generator — generador aleatorio.

    Retorna
    -------
    np.ndarray — población de padres seleccionados, shape (n_offspring, k).

    Nota
    ----
    Esta función NO llama a la función objetivo: opera sobre ranks y crowding
    ya calculados, por lo que no consume presupuesto de evaluaciones.
    """
    pop_size = population.shape[0]
    selected = np.empty((n_offspring, population.shape[1]), dtype=population.dtype)

    for i in range(n_offspring):
        a, b = rng.integers(0, pop_size), rng.integers(0, pop_size)
        # Gana el de mejor (menor) rank; si empatan, mayor crowding distance.
        if ranks[a] < ranks[b]:
            winner = a
        elif ranks[b] < ranks[a]:
            winner = b
        else:
            winner = a if crowding[a] >= crowding[b] else b
        selected[i] = population[winner]

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 3. UTILIDADES DE FRENTE
# ══════════════════════════════════════════════════════════════════════════════

def _extract_f1(
    population: np.ndarray,
    objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae el frente no dominado F1 de una población ya evaluada.

    Parámetros
    ----------
    population : np.ndarray — población, shape (m, k).
    objectives : np.ndarray — objetivos (XBEB, XBBB), shape (m, 2).

    Retorna
    -------
    tuple (f1_population, f1_objectives) con solo las soluciones de F1.

    Nota
    ----
    No consume presupuesto: opera sobre objetivos ya calculados.
    """
    ranks, _, fronts = non_dominated_sorting_crowding_distance(objectives)
    f1_idx = fronts[0]
    return population[f1_idx], objectives[f1_idx]


def _dedup(population: np.ndarray) -> np.ndarray:
    """
    Elimina soluciones duplicadas (mismo conjunto de medoides) de una población.

    Parámetros
    ----------
    population : np.ndarray — población, shape (m, k).

    Retorna
    -------
    np.ndarray — población sin duplicados.
    """
    seen = set()
    unique = []
    for sol in population:
        key = frozenset(sol.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(sol)
    return np.stack(unique)


# ══════════════════════════════════════════════════════════════════════════════
# 4. HIPERVOLUMEN (indicador de convergencia)
# ══════════════════════════════════════════════════════════════════════════════

def hypervolume(
    objectives: np.ndarray,
    reference_point: tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    Calcula el hipervolumen (HV) de un frente de Pareto para 2 objetivos
    de minimización, según la ecuación (7) del paper.

    El HV es el área (en 2D) dominada por el frente respecto a un punto de
    referencia W. Como ambos objetivos se minimizan, cada solución i cubre
    un rectángulo [XBEB_i, W_x] × [XBBB_i, W_y]; el HV es el área de la
    unión de todos esos rectángulos. Valores mayores indican un frente que
    cubre más espacio objetivo (mejor convergencia y dispersión).

    Las soluciones con valores no finitos (XB=inf, degeneradas) o que estén
    fuera del punto de referencia (peor que W en algún objetivo) se ignoran.

    Parámetros
    ----------
    objectives      : np.ndarray — frente, shape (m, 2) con (XBEB, XBBB).
    reference_point : tuple[float, float] — punto de referencia W. El paper
                      usa (1, 1) como referencia normalizada. Default (1, 1).

    Retorna
    -------
    float — valor del hipervolumen. 0.0 si no hay soluciones válidas dentro
            del punto de referencia.

    Ejemplo
    -------
    >>> hv = hypervolume(f1_objectives, reference_point=(1.0, 1.0))
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    if objectives.ndim != 2 or objectives.shape[1] != 2:
        raise ValueError(f"Se esperaba shape (m, 2), se recibió {objectives.shape}.")

    ref_x, ref_y = reference_point

    # Filtrar: finitas y estrictamente dentro del punto de referencia.
    finite_mask = np.all(np.isfinite(objectives), axis=1)
    within_mask = (objectives[:, 0] < ref_x) & (objectives[:, 1] < ref_y)
    valid = objectives[finite_mask & within_mask]

    if len(valid) == 0:
        return 0.0

    # Conservar solo el frente no dominado dentro de los puntos válidos
    # (puntos dominados no aportan área a la unión de rectángulos).
    # Ordenar por XBEB ascendente; ante empate, por XBBB ascendente.
    order = np.lexsort((valid[:, 1], valid[:, 0]))
    valid = valid[order]

    # Barrido para minimización con referencia W (esquina superior-derecha):
    # recorriendo por XBEB creciente, una solución solo aporta área nueva si
    # su XBBB es menor que el mejor (menor) XBBB visto hasta ahora. El ancho
    # de su franja va desde su XBEB hasta el XBEB de la siguiente solución no
    # dominada (o hasta ref_x si es la última).
    nd_points = []
    best_y = np.inf
    for x, y in valid:
        if y < best_y:
            nd_points.append((x, y))
            best_y = y

    hv = 0.0
    for i, (x, y) in enumerate(nd_points):
        # Límite derecho de la franja: XBEB del siguiente punto no dominado,
        # o ref_x si es el último.
        x_next = nd_points[i + 1][0] if i + 1 < len(nd_points) else ref_x
        hv += (x_next - x) * (ref_y - y)

    return hv


# ══════════════════════════════════════════════════════════════════════════════
# 5. ALGORITMO PRINCIPAL MOC-GaPBK
# ══════════════════════════════════════════════════════════════════════════════

def moc_gapbk(
    deb_matrix: np.ndarray,
    dbb_matrix: np.ndarray,
    k: int,
    pop_size: int,
    max_obj_calls: int,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    pls_max_iterations: int = 100,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta la metaheurística MOC-GaPBK completa.

    El flujo replica el Algoritmo 1 del paper, pero el criterio de paro es
    un máximo de llamadas a la función objetivo (max_obj_calls), donde cada
    índice Xie-Beni evaluado cuenta como 1 llamada.

    Parámetros
    ----------
    deb_matrix         : np.ndarray — matriz (n×n) de distancia de expresión (DEB).
    dbb_matrix         : np.ndarray — matriz (n×n) de distancia biológica (DBB).
    k                  : int        — número de clusters.
    pop_size           : int        — tamaño de la población N.
    max_obj_calls      : int        — máximo de llamadas a la función objetivo.
    crossover_prob     : float      — probabilidad de crossover (default 0.80).
    mutation_prob      : float      — probabilidad de mutación (default 0.01).
    pls_max_iterations : int        — iteraciones máximas de PLS por generación.
    seed               : int        — semilla para reproducibilidad.
    verbose            : bool       — si True, imprime el progreso por generación.

    Retorna
    -------
    dict con claves:
        "pareto_front"        : np.ndarray — soluciones no dominadas finales (F1).
        "pareto_objectives"   : np.ndarray — (XBEB, XBBB) de F1, shape (|F1|, 2).
        "fronts"              : list[list[int]] — frentes de la población final.
        "generations"         : int — número de generaciones completadas.
        "objective_calls"     : int — total de llamadas a función objetivo usadas.
        "hv_history"          : list[float] — hipervolumen del frente F1 en
                                cada generación (insumo del gráfico de convergencia).

    Ejemplo
    -------
    >>> result = moc_gapbk(deb, dbb, k=4, pop_size=50, max_obj_calls=20000, seed=42)
    >>> print(result["pareto_front"])
    """
    rng = np.random.default_rng(seed)
    n = deb_matrix.shape[0]

    counter = ObjectiveCounter(max_calls=max_obj_calls)
    counter.patch()

    try:
        # ── Línea 4: Población inicial P0 ───────────────────────────────────
        Pv = random_medoids_pop(n=n, k=k, pop_size=pop_size, seed=seed)
        Pv_obj = pareto_sorting.evaluate_population_objectives(Pv, deb_matrix, dbb_matrix)

        generation = 0
        hv_history = []  # historial de hipervolumen del frente F1 por generación

        if verbose:
            print(f"[MOC-GaPBK] Inicio. n={n}, k={k}, N={pop_size}, "
                  f"presupuesto={max_obj_calls} llamadas.")
            print(f"[MOC-GaPBK] P0 evaluada. Llamadas usadas: {counter.count}.")

        # ── Línea 5: bucle principal ────────────────────────────────────────
        while not counter.budget_exhausted():
            generation += 1

            # Rank + crowding de la población actual (para selección).
            ranks, crowding, _ = non_dominated_sorting_crowding_distance(Pv_obj)

            # ── Líneas 6-7: selección + offspring (crossover + mutación) ────
            parents = binary_tournament_selection(
                Pv, ranks, crowding, n_offspring=pop_size, rng=rng
            )
            Qv = apply_operators_population(
                parents, n=n,
                crossover_prob=crossover_prob, mutation_prob=mutation_prob,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            Qv_obj = pareto_sorting.evaluate_population_objectives(Qv, deb_matrix, dbb_matrix)

            # ── Línea 8: Rv = Pv ∪ Qv ───────────────────────────────────────
            Rv = _dedup(np.vstack([Pv, Qv]))
            Rv_obj = pareto_sorting.evaluate_population_objectives(Rv, deb_matrix, dbb_matrix)

            # ── Líneas 9-10: non-dominated sorting → F1 ─────────────────────
            f1_pop, f1_obj = _extract_f1(Rv, Rv_obj)

            # ── Líneas 11-12: Intensificación (MOPR sobre pares de F1) ─────
            if not counter.budget_exhausted() and len(f1_pop) >= 2:
                # Emparejamos las soluciones de F1 secuencialmente y aplicamos
                # MOPR a cada par; fusionamos todos los F1 resultantes.
                mopr_solutions = [f1_pop]
                for p in range(len(f1_pop) - 1):
                    if counter.budget_exhausted():
                        break
                    C1, C2 = f1_pop[p], f1_pop[p + 1]
                    if frozenset(C1.tolist()) == frozenset(C2.tolist()):
                        continue
                    mopr_res = multi_objective_path_relinking(C1, C2, deb_matrix, dbb_matrix)
                    mopr_solutions.append(mopr_res["f1"])

                merged = _dedup(np.vstack(mopr_solutions))
                merged_obj = pareto_sorting.evaluate_population_objectives(merged, deb_matrix, dbb_matrix)
                f1_pop, f1_obj = _extract_f1(merged, merged_obj)

            # ── Líneas 13-14: Diversificación (PLS sobre F1) ───────────────
            if not counter.budget_exhausted() and len(f1_pop) >= 1:
                pls_pop, pls_obj_list = pareto_local_search(
                    initial_population=f1_pop,
                    deb_matrix=deb_matrix, dbb_matrix=dbb_matrix,
                    max_iterations=pls_max_iterations,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                pls_obj = np.asarray(pls_obj_list, dtype=np.float64)
                combined = _dedup(np.vstack([f1_pop, pls_pop]))
                combined_obj = pareto_sorting.evaluate_population_objectives(combined, deb_matrix, dbb_matrix)
                f1_pop, f1_obj = _extract_f1(combined, combined_obj)

            # ── Líneas 15-21: construir Pv+1 ────────────────────────────────
            if len(f1_pop) >= pop_size:
                # Si F1 excede N, truncar por crowding distance.
                ranks_f1, crowding_f1, _ = non_dominated_sorting_crowding_distance(f1_obj)
                order = np.argsort(-crowding_f1)  # mayor crowding primero
                keep = order[:pop_size]
                Pv = f1_pop[keep]
                Pv_obj = f1_obj[keep]
            else:
                # Completar con población aleatoria (línea 17-18).
                n_random = pop_size - len(f1_pop)
                Pr = random_medoids_pop(
                    n=n, k=k, pop_size=n_random,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                Pr_obj = pareto_sorting.evaluate_population_objectives(Pr, deb_matrix, dbb_matrix)
                Pv = np.vstack([f1_pop, Pr])
                Pv_obj = np.vstack([f1_obj, Pr_obj])

            # ── Registro de convergencia: hipervolumen del frente F1 ────────
            hv_gen = hypervolume(f1_obj, reference_point=(1.0, 1.0))
            hv_history.append(hv_gen)

            if verbose:
                print(f"[MOC-GaPBK] Gen {generation:>3} | "
                      f"|F1|={len(f1_pop):>3} | "
                      f"llamadas={counter.count:>7}/{max_obj_calls} | "
                      f"HV={hv_gen:.6f} | "
                      f"mejor XBEB={f1_obj[:, 0].min():.4f}, "
                      f"mejor XBBB={f1_obj[:, 1].min():.4f}")

    finally:
        counter.restore()

    # ── Línea 24: frente de Pareto final ────────────────────────────────────
    final_ranks, _, final_fronts = non_dominated_sorting_crowding_distance(Pv_obj)
    pareto_idx = final_fronts[0]
    pareto_front = Pv[pareto_idx]
    pareto_objectives = Pv_obj[pareto_idx]

    return {
        "pareto_front": pareto_front,
        "pareto_objectives": pareto_objectives,
        "fronts": final_fronts,
        "final_population": Pv,
        "final_objectives": Pv_obj,
        "generations": generation,
        "objective_calls": counter.count,
        "hv_history": hv_history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIÓN DE CONVERGENCIA (hipervolumen por generación)
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(
    hv_history: list[float],
    output_dir: str | None = None,
    filename: str = "convergence.png",
    title: str = "Convergencia de MOC-GaPBK (hipervolumen por generación)",
    show: bool = False,
) -> str | None:
    """
    Grafica la curva de convergencia de la metaheurística: el hipervolumen
    del frente de Pareto F1 en función del número de generación.

    Un hipervolumen creciente y que se estabiliza (mesetea) indica que el
    algoritmo converge: el frente cubre cada vez más espacio objetivo hasta
    dejar de mejorar significativamente.

    Parámetros
    ----------
    hv_history : list[float] — hipervolumen por generación, salida de
                 moc_gapbk()["hv_history"].
    output_dir : str — carpeta donde guardar el gráfico (se crea si no
                 existe). Si es None, no se guarda archivo.
    filename   : str — nombre del archivo de imagen.
    title      : str — título del gráfico.
    show       : bool — si True, muestra el gráfico interactivamente.

    Retorna
    -------
    str | None — ruta del archivo guardado, o None si output_dir es None.

    Ejemplo
    -------
    >>> plot_convergence(result["hv_history"], output_dir="resultados")
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generations = np.arange(1, len(hv_history) + 1)
    hv = np.asarray(hv_history, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(generations, hv, marker="o", color="#2a6f97",
            linewidth=1.8, markersize=6, markeredgecolor="black",
            markeredgewidth=0.5, zorder=3)
    ax.fill_between(generations, hv, alpha=0.15, color="#2a6f97", zorder=1)

    # Marcar el mejor hipervolumen alcanzado.
    if len(hv) > 0:
        best_gen = int(np.argmax(hv)) + 1
        best_hv = float(np.max(hv))
        ax.scatter([best_gen], [best_hv], s=160, facecolors="none",
                   edgecolors="#e63946", linewidths=2.2, zorder=4,
                   label=f"Mejor HV = {best_hv:.5f} (gen {best_gen})")
        ax.legend(loc="lower right", fontsize=10)

    ax.set_xlabel("Generación", fontsize=12)
    ax.set_ylabel("Hipervolumen (HV) del frente F1", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    if len(generations) > 0:
        ax.set_xlim(0.5, len(generations) + 0.5)
        ax.set_xticks(generations if len(generations) <= 20
                      else np.linspace(1, len(generations), 20, dtype=int))
    fig.tight_layout()

    saved_path = None
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_path = out_path / filename
        fig.savefig(saved_path, dpi=150)
        print(f"[save] Gráfico de convergencia guardado en: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(saved_path) if saved_path is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE COMPLETO (carga, ejecución, guardado de resultados e imagen)
# ══════════════════════════════════════════════════════════════════════════════

def run_moc_gapbk(
    deb_path: str,
    dbb_path: str,
    k: int,
    pop_size: int,
    max_obj_calls: int,
    output_dir: str,
    crossover_prob: float = 0.80,
    mutation_prob: float = 0.01,
    pls_max_iterations: int = 100,
    seed: int | None = None,
    prefix: str = "moc_gapbk",
    verbose: bool = True,
) -> dict:
    """
    Pipeline completo de MOC-GaPBK: carga matrices → ejecuta el algoritmo →
    guarda la solución (objetivos + clusters) y la IMAGEN FINAL de las
    fronteras de Pareto.

    Parámetros
    ----------
    deb_path           : str   — ruta a la matriz DEB (.csv).
    dbb_path           : str   — ruta a la matriz DBB (.csv).
    k                  : int   — número de clusters.
    pop_size           : int   — tamaño de la población N.
    max_obj_calls      : int   — máximo de llamadas a función objetivo.
    output_dir         : str   — carpeta de salida.
    crossover_prob     : float — probabilidad de crossover (default 0.80).
    mutation_prob      : float — probabilidad de mutación (default 0.01).
    pls_max_iterations : int   — iteraciones máximas de PLS por generación.
    seed               : int   — semilla para reproducibilidad.
    prefix             : str   — prefijo para los archivos de salida.
    verbose            : bool  — si True, imprime el progreso.

    Retorna
    -------
    dict con las claves de moc_gapbk() más:
        "gene_names"      : list[str].
        "objectives_path" : str — CSV con medoides + XBEB/XBBB del frente final.
        "clusters_path"   : str — CSV con la asignación de clusters del frente final.
        "plot_path"       : str — imagen final de las fronteras de Pareto.
        "convergence_path": str — gráfico de convergencia (hipervolumen por generación).

    Ejemplo
    -------
    >>> output = run_moc_gapbk(
    ...     deb_path="data/DEB.csv", dbb_path="data/DBB.csv",
    ...     k=4, pop_size=50, max_obj_calls=20000,
    ...     output_dir="resultados", seed=42,
    ... )
    """
    # ── Carga de matrices ───────────────────────────────────────────────────
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

    # ── Ejecución del algoritmo ─────────────────────────────────────────────
    result = moc_gapbk(
        deb_matrix=deb, dbb_matrix=dbb,
        k=k, pop_size=pop_size, max_obj_calls=max_obj_calls,
        crossover_prob=crossover_prob, mutation_prob=mutation_prob,
        pls_max_iterations=pls_max_iterations,
        seed=seed, verbose=verbose,
    )

    pareto_front = result["pareto_front"]
    pareto_obj = result["pareto_objectives"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Tabla de objetivos del frente final ────────────────────────────────
    rows = []
    for i, (medoids, (xbeb, xbbb)) in enumerate(zip(pareto_front, pareto_obj)):
        row = {"solution_id": f"pareto_{i + 1}"}
        for m in range(k):
            row[f"medoid_{m + 1}"] = gene_names[medoids[m]]
        row["XBEB"] = xbeb
        row["XBBB"] = xbbb
        rows.append(row)
    df_obj = pd.DataFrame(rows)
    objectives_path = out_path / f"{prefix}_pareto_objectives.csv"
    df_obj.to_csv(objectives_path, index=False)
    print(f"[save] Objetivos del frente de Pareto guardados en: {objectives_path}")

    # ── Asignación de clusters del frente final (con DEB) ──────────────────
    all_labels = build_clusters_population(deb, pareto_front)
    clusters_path = save_clusters(
        all_labels=all_labels,
        gene_names=gene_names,
        output_dir=output_dir,
        filename=f"{prefix}_pareto_clusters.csv",
    )

    # ── IMAGEN FINAL de las fronteras de Pareto ────────────────────────────
    plot_path = plot_pareto_fronts(
        result["final_objectives"],
        fronts=result["fronts"],
        output_dir=output_dir,
        filename=f"{prefix}_pareto_fronts_final.png",
        title=f"MOC-GaPBK — Frente de Pareto final\n"
              f"({result['generations']} generaciones, "
              f"{result['objective_calls']} llamadas a función objetivo)",
    )

    # ── GRÁFICO DE CONVERGENCIA (hipervolumen por generación) ──────────────
    convergence_path = plot_convergence(
        result["hv_history"],
        output_dir=output_dir,
        filename=f"{prefix}_convergence.png",
        title="MOC-GaPBK — Convergencia (hipervolumen del frente F1 por generación)",
    )

    if verbose:
        print(f"\n[MOC-GaPBK] Finalizado.")
        print(f"  Generaciones completadas : {result['generations']}")
        print(f"  Llamadas a func. objetivo: {result['objective_calls']} / {max_obj_calls}")
        print(f"  Soluciones en frente final: {len(pareto_front)}")
        if result["hv_history"]:
            print(f"  Hipervolumen final        : {result['hv_history'][-1]:.6f}")

    return {
        **result,
        "gene_names": gene_names,
        "objectives_path": str(objectives_path),
        "clusters_path": clusters_path,
        "plot_path": plot_path,
        "convergence_path": convergence_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    deb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBE.csv"
    dbb_path   = r"C:\Users\benja\Desktop\workspace\Thesis\Datasets\DGE_DBI\GSE40419_DBI.csv"
    output_dir = r"C:\Users\benja\Desktop\workspace\Thesis\Results"

    output = run_moc_gapbk(
        deb_path=deb_path,
        dbb_path=dbb_path,
        k=4,
        pop_size=100,
        max_obj_calls=80000,
        output_dir=output_dir,
        crossover_prob=0.80,
        mutation_prob=0.2,
        pls_max_iterations=200,
        seed=40,
        prefix="moc_gapbk",
    )

    print("\n── Frente de Pareto final ──")
    for sol, (xbeb, xbbb) in zip(output["pareto_front"], output["pareto_objectives"]):
        genes = [output["gene_names"][i] for i in sol]
        print(f"  {genes}  → XBEB={xbeb:.6f}, XBBB={xbbb:.6f}")