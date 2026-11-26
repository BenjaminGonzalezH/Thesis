#######################################################
# MODULO DE EJECUCIÓN EN PARALELO (Ejecuciones.ipynb)
# Una corrida = variante × dataset × semilla. Cada worker (proceso `spawn`) carga
# el dataset bajo demanda y lo conserva en memoria mientras las tareas siguientes
# sean del mismo dataset. El worker guarda el .npz del frente y DEVUELVE la fila del
# historial; el CSV lo escribe solo el proceso padre (sin escrituras concurrentes).
#######################################################

import io
import json
import time
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np

import solution_encoding                                    # OBJ_FUNCTION_CALLS es global POR PROCESO (spawn): no se comparte entre workers.
from solution_encoding import load_distance_matrix
from LS import build_neighborhood_matrix
from nsga2 import run_nsga2
from performance import jaccard_f0_history, jaccard_population_history
from nsga2_variants import run_nsga2_MOPR, run_nsga2_PLS, run_nsga2_MOLS
from nsga2_variants_bio import run_nsga2_PNS, run_nsga2_DBS, run_nsga2_MPS

VARIANTES = {
    "NSGA2":       run_nsga2,
    "NSGA2_PLS":   run_nsga2_PLS,
    "NSGA2_MOLS":  run_nsga2_MOLS,
    "NSGA2_DBS":   run_nsga2_DBS,
    "NSGA2_PNS":   run_nsga2_PNS,
    "NSGA2_MPS":   run_nsga2_MPS,
    "NSGA2_MOPR":  run_nsga2_MOPR,
}

_CACHE = {}                                                 # Por proceso: un solo dataset en memoria a la vez.


def cargar_dataset(matrices_dir, nombre: str) -> dict:
    """Carga GE/BI. Los M_V y sus radios se calculan solo cuando una configuración de MOLS los requiere."""
    if nombre not in _CACHE:
        _CACHE.clear()
        ge, _ = load_distance_matrix(Path(matrices_dir) / f"{nombre}_S_DE.npz")
        bi, _ = load_distance_matrix(Path(matrices_dir) / f"{nombre}_S_DB_M.npz")
        _CACHE[nombre] = {"ge": ge, "bi": bi, "n": ge.shape[0], "m_v": {}, "radios": {}}
    return _CACHE[nombre]


def radio_mols(ds: dict, fuente: str, percentil: int) -> tuple:
    """M_V y radio `neighborhood` (percentil de M_V sin diagonal), con memoización por dataset."""
    if fuente not in ds["m_v"]:
        ds["m_v"][fuente] = build_neighborhood_matrix(ds["ge"], ds["bi"], fuente)
    clave = (fuente, percentil)
    if clave not in ds["radios"]:
        m_v = ds["m_v"][fuente]
        ds["radios"][clave] = float(np.percentile(m_v[~np.eye(ds["n"], dtype=bool)], percentil))
    return ds["m_v"][fuente], ds["radios"][clave]


def kwargs_variante(variante: str, cfg: dict, ds: dict, seed: int, k: int, max_calls: int) -> dict:
    kw = dict(
        ge_matrix=ds["ge"], bi_matrix=ds["bi"], n=ds["n"], k=k,
        pop_size=cfg["pop_size"], max_obj_function_calls=max_calls,
        crossover_prob=cfg["crossover_prob"], mutation_prob=cfg["mutation_prob"],
        seed=seed,
    )
    if variante == "NSGA2_PLS":
        kw["pls_neighborhood"] = cfg["pls_neighborhood"]
        kw["pls_local_budget"] = cfg["pls_local_budget"]
    elif variante == "NSGA2_MOLS":
        m_v, radio = radio_mols(ds, cfg["M_v"], cfg["neighborhood_pct"])
        kw["mode"]              = cfg["mode"]
        kw["m_v_matrix"]        = m_v
        kw["neighborhood"]      = radio
        kw["mols_local_budget"] = cfg["mols_local_budget"]
    elif variante == "NSGA2_PNS":
        kw["nF"]               = cfg["nF"]
    elif variante == "NSGA2_DBS":
        kw["mode"]             = cfg["mode"]
        kw["p"]                = cfg["p_pct"] / 100
    elif variante == "NSGA2_MPS":
        kw["L"]                = cfg["L"]
        kw["q"]                = cfg["q_pct"] / 100
    return kw


def ejecutar(variante: str, nombre_ds: str, semilla: int, cfg: dict,
             matrices_dir, frentes_dir, k: int, max_calls: int) -> dict:
    """Una corrida completa. Guarda el .npz del frente F0 y devuelve la fila del historial."""
    ds = cargar_dataset(matrices_dir, nombre_ds)

    solution_encoding.OBJ_FUNCTION_CALLS = 0                # Reinicio obligatorio del presupuesto.
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):         # Se silencia el print por generación.
        res = VARIANTES[variante](**kwargs_variante(variante, cfg, ds, semilla, k, max_calls))
    segundos = time.time() - t0

    hv = res["history_hv_f1"]
    jac = jaccard_f0_history(res["history_objectives"], res["history_labels"])
    jac_pop = jaccard_population_history(res["history_labels"])
    f0 = np.asarray(res["fronts"][0])
    np.savez_compressed(
        Path(frentes_dir) / f"{variante}_{nombre_ds}_{semilla}.npz",
        objectives=np.asarray(res["objectives"])[f0],
        medoids=np.asarray(res["population"])[f0],
        history_hv_f1=np.asarray(hv, dtype=float),
        history_jaccard_f0=np.asarray(jac, dtype=float),
        history_jaccard_pop=np.asarray(jac_pop, dtype=float),
    )
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "variante": variante, "dataset": nombre_ds, "semilla": semilla,
        "hv_inicial": float(hv[0]), "hv_final": float(hv[-1]),
        "tam_f0": int(len(f0)), "generaciones": int(res["generations_run"]),
        "llamadas": int(solution_encoding.OBJ_FUNCTION_CALLS), "segundos": round(segundos, 1),
        "parametros": json.dumps(cfg, ensure_ascii=False),
    }
