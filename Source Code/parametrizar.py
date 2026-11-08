#######################################################
# PARAMETRIZACIÓN CON OPTUNA — ejecución de UNA variante en su propio proceso
#
# Parametrización.ipynb lanza un proceso por variante (estudios independientes, un .db por
# variante). Las funciones viven en este módulo (y no en el notebook) para que puedan
# serializarse hacia los procesos hijos (Windows usa `spawn`).
#
# Por qué procesos y no hilos: solution_encoding.OBJ_FUNCTION_CALLS es un contador global
# (cuota exacta de llamadas) y redirect_stdout también es global al proceso.
#######################################################

import os

# BLAS/OpenMP de un solo hilo por proceso: el paralelismo ya se hace entre variantes.
# Debe fijarse ANTES de importar numpy (en el proceso hijo se importa este módulo primero).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import io
import sys
import time
import contextlib
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

import solution_encoding                                    # OBJ_FUNCTION_CALLS se lee/reinicia como solution_encoding.OBJ_FUNCTION_CALLS.
from solution_encoding import load_distance_matrix
from LS import build_neighborhood_matrix
from nsga2 import run_nsga2
from nsga2_variants import run_nsga2_MOPR, run_nsga2_PLS, run_nsga2_MOLS
from nsga2_variants_bio import run_nsga2_PNS, run_nsga2_DBS, DBS_MODES

############################
# Constantes del experimento (según la tabla de parametrización)
############################
DATASETS                 = ("A1", "A2")                  # Datasets de parametrización (prefijo A).
K                        = 4                             # Medoides por solución.
MAX_OBJ_FUNCTION_CALLS   = 400_000                       # Presupuesto POR ejecución y POR dataset.
PERCENTILES_NEIGHBORHOOD = range(10, 36)                 # Percentil de M_V para `neighborhood`: [10, 35].
MV_OPCIONES              = ("GE", "BI", "BOTH")          # Matrices que pueden definir M_V (vecindario de MOLS).
MOLS_MODES               = ("large", "narrow")           # Criterio de aceptación de MOLS: L-MOLS / N-MOLS.

FUNCIONES_OPTUNA = {
    "NSGA2":        run_nsga2,
    "NSGA2_PLS":    run_nsga2_PLS,
    "NSGA2_MOLS":   run_nsga2_MOLS,
    "NSGA2_DBS":    run_nsga2_DBS,
    "NSGA2_PNS":    run_nsga2_PNS,
    "NSGA2_MOPR":   run_nsga2_MOPR,
}


def storage_url(results_dir, nombre_variante: str) -> str:
    """Un archivo .db por variante: estudios independientes y fáciles de respaldar."""
    return f"sqlite:///{Path(results_dir) / (nombre_variante + '.db')}"


def crear_sampler(seed: int) -> TPESampler:
    """Mismo muestreador y misma semilla para TODAS las variantes (la diferencia entre estudios proviene del algoritmo)."""
    return TPESampler(seed=seed, multivariate=True)         # n_startup_trials: valor por defecto de Optuna.


############################
# Carga de datasets
############################
def cargar_dataset(matrices_dir, nombre: str) -> dict:
    """Carga GE/BI y precalcula, para cada opción de M_V, los radios `neighborhood` (percentiles de M_V, sin diagonal)."""
    matrices_dir = Path(matrices_dir)
    ge, _ = load_distance_matrix(matrices_dir / f"{nombre}_S_DE.npz")
    bi, _ = load_distance_matrix(matrices_dir / f"{nombre}_S_DB_M.npz")
    n = ge.shape[0]

    m_v, radios = {}, {}
    for fuente in MV_OPCIONES:
        m_v[fuente] = build_neighborhood_matrix(ge, bi, fuente)
        valores = m_v[fuente][~np.eye(n, dtype=bool)]       # M_V es simétrica: los duplicados no alteran el percentil.
        radios[fuente] = dict(zip(
            PERCENTILES_NEIGHBORHOOD,
            np.percentile(valores, list(PERCENTILES_NEIGHBORHOOD)).astype(float),
        ))
        del valores

    return {"ge": ge, "bi": bi, "m_v": m_v, "n": n, "neighborhood": radios}


############################
# Función objetivo
############################
def _sugerir_configuracion(trial: optuna.Trial, variante: str) -> dict:
    """Muestrea los hiperparámetros de la variante (rangos de la tabla). Es UNA configuración para ambos datasets."""
    cfg = {
        "pop_size":       trial.suggest_int("pop_size", 50, 300, step=1),
        "crossover_prob": round(trial.suggest_float("crossover_prob", 0.80, 0.90, step=0.01), 2),
        "mutation_prob":  round(trial.suggest_float("mutation_prob",  0.01, 0.05, step=0.01), 2),
    }
    if variante == "NSGA2_PLS":
        cfg["pls_neighborhood"]     = round(trial.suggest_float("pls_neighborhood", 0.05, 0.25, step=0.01), 2)
        cfg["pls_local_budget"]     = trial.suggest_int("pls_local_budget", 1000, 10000, step=100)
    elif variante == "NSGA2_MOLS":
        cfg["M_v"]                   = trial.suggest_categorical("M_v", list(MV_OPCIONES))
        cfg["mode"]                  = trial.suggest_categorical("mode", list(MOLS_MODES))
        cfg["mols_local_budget"]     = trial.suggest_int("mols_local_budget", 1000, 10000, step=100)
        cfg["neighborhood_pct"]      = trial.suggest_int("neighborhood_pct", 10, 35)
    elif variante == "NSGA2_PNS":
        cfg["nF"]                   = trial.suggest_int("nF", 1, 5)
    elif variante == "NSGA2_DBS":
        cfg["mode"]                 = trial.suggest_categorical("mode", list(DBS_MODES))
        cfg["p_pct"]                = trial.suggest_int("p_pct", 5, 25)
    return cfg


def _kwargs_variante(variante: str, cfg: dict, ds: dict, seed: int, max_calls: int) -> dict:
    """Traduce la configuración muestreada a los argumentos reales de la función de la variante."""
    kw = dict(
        ge_matrix=ds["ge"], bi_matrix=ds["bi"], n=ds["n"], k=K,
        pop_size=cfg["pop_size"], max_obj_function_calls=max_calls,
        crossover_prob=cfg["crossover_prob"], mutation_prob=cfg["mutation_prob"],
        seed=seed,
    )
    if variante == "NSGA2_PLS":
        kw["pls_neighborhood"] = cfg["pls_neighborhood"]
        kw["pls_local_budget"] = cfg["pls_local_budget"]
    elif variante == "NSGA2_MOLS":
        kw["mode"]              = cfg["mode"]
        kw["m_v_matrix"]        = ds["m_v"][cfg["M_v"]]
        kw["neighborhood"]      = ds["neighborhood"][cfg["M_v"]][cfg["neighborhood_pct"]]
        kw["mols_local_budget"] = cfg["mols_local_budget"]
    elif variante == "NSGA2_PNS":
        kw["nF"]               = cfg["nF"]
    elif variante == "NSGA2_DBS":
        kw["mode"]             = cfg["mode"]
        kw["p"]                = cfg["p_pct"] / 100                 # Percentil p -> fracción.
    return kw


def objective(trial: optuna.Trial, variante: str, datos: dict, seeds_param: list, max_calls: int) -> float:
    """
    Retorna el HIPERVOLUMEN FINAL de F0 (respecto a HV_REFERENCE_POINT), promediado
    entre datasets (se MAXIMIZA). Cada trial usa UNA semilla de `seeds_param`, según su
    número (trial.number): como la lista es la misma para todas las variantes, el
    trial N de cada una se evalúa con la misma semilla. Ambos datasets comparten la
    semilla del trial (2 ejecuciones por trial).
    """
    cfg     = _sugerir_configuracion(trial, variante)
    semilla = seeds_param[trial.number % len(seeds_param)]
    trial.set_user_attr("semilla", semilla)
    hvs = []

    for nombre, ds in datos.items():
        solution_encoding.OBJ_FUNCTION_CALLS = 0            # Reinicio obligatorio: el presupuesto NO se comparte entre datasets.
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):     # Se silencia el print por generación.
            resultado = FUNCIONES_OPTUNA[variante](**_kwargs_variante(variante, cfg, ds, semilla, max_calls))

        hv = resultado["history_hv_f1"]
        hvs.append(hv[-1])

        trial.set_user_attr(f"{nombre}_hv_f1_inicial", float(hv[0]))
        trial.set_user_attr(f"{nombre}_hv_f1_final", float(hv[-1]))
        trial.set_user_attr(f"{nombre}_generaciones", int(resultado["generations_run"]))
        trial.set_user_attr(f"{nombre}_llamadas", int(solution_encoding.OBJ_FUNCTION_CALLS))
        trial.set_user_attr(f"{nombre}_segundos", round(time.time() - t0, 1))

    return float(np.mean(hvs))


############################
# Ejecución de una variante (punto de entrada de cada proceso)
############################
def ejecutar_variante(
    variante: str,
    n_trials: int,
    seed: int,
    seeds_param: list,
    results_dir,
    matrices_dir,
    max_calls: int = MAX_OBJ_FUNCTION_CALLS,
    datasets=DATASETS,
) -> dict:
    """
    Crea/reanuda el estudio de `variante` (SQLite en results_dir) y completa los trials faltantes
    hasta `n_trials`. El log va a results_dir/log_<variante>.txt. Retorna el mejor valor y parámetros.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results_dir = Path(results_dir)
    log_path    = results_dir / f"log_{variante}.txt"

    def log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    datos   = {nombre: cargar_dataset(matrices_dir, nombre) for nombre in datasets}
    estudio = optuna.create_study(
        study_name=variante,
        storage=storage_url(results_dir, variante),
        sampler=crear_sampler(seed),
        direction="maximize",                               # Se maximiza el hipervolumen final de F0.
        load_if_exists=True,                                # Permite reanudar un estudio interrumpido.
    )

    completados = sum(t.state == optuna.trial.TrialState.COMPLETE for t in estudio.trials)
    pendientes  = max(0, n_trials - completados)
    log(f"{variante}: {completados} completados, {pendientes} pendientes (total {n_trials})")

    def _log_trial(est: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        segundos = sum(v for k, v in trial.user_attrs.items() if k.endswith("_segundos"))
        log(f"trial {trial.number:>3} -> HV F0 = {trial.value:.4f} ({segundos / 60:.1f} min) {trial.params}")

    if pendientes:
        estudio.optimize(
            lambda t: objective(t, variante, datos, seeds_param, max_calls),
            n_trials=pendientes, callbacks=[_log_trial],
        )

    return {"variante": variante, "hv_f1": estudio.best_value, "parametros": estudio.best_params}
