"""
Visualización y métricas de desempeño: frentes de Pareto, diversidad
(Jaccard) e hipervolumen de F0 respecto a un punto de referencia.
"""

############################
# Importaciones
############################
from pathlib import Path                # Administración global de directorio de archivo.
import numpy as np                      # Operaciones matemáticas eficientes.
import matplotlib.pyplot as plt         # Gráficos personalizados.

##################################
# Configuraciones
##################################
# Se da opción de ocupar GPU para administrar operaciones vectorizadas
# y operadas por ese hardware en caso de ser necesario.
try:
    import cupy as cp
    _GPU = True
    print("[backend] CuPy detectado → operaciones vectorizadas en GPU.")
except ImportError:
    cp = np
    _GPU = False
    print("[backend] CuPy no encontrado → usando NumPy (CPU).")

##################################
# Conjunto de funciones.
##################################

def plot_pareto_fronts(
    objectives,
    fronts: list[list[int]],
    output_dir: str | None = None,
    filename: str = "pareto_fronts.png",
    title: str = "Frentes de Pareto",
    max_fronts_highlighted: int = 5,
    show: bool = False,
) -> str | None:
    """
    Grafica los frentes de Pareto (XB_GE vs XB_BI). Soluciones con XB=inf
    (degeneradas) se excluyen del gráfico.
    """
    XLABEL = "XB_GE (Expresión)"
    YLABEL = "XB_BI (Biológico)"

    objectives = np.asarray(objectives, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    n_fronts = len(fronts)

    for rank, front in enumerate(fronts):
        pts = objectives[front]
        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts_finite = pts[finite_mask]

        if len(pts_finite) == 0:
            continue

        color = cmap(rank / max(n_fronts - 1, 1))
        label = f"F{rank}" if rank < max_fronts_highlighted else None

        ax.scatter(
            pts_finite[:, 0], pts_finite[:, 1],
            color=color, label=label,
            s=45, edgecolors="black", linewidths=0.5,
            zorder=n_fronts - rank,
        )

        if rank == 0 and len(pts_finite) > 1:
            order = np.argsort(pts_finite[:, 0])
            ax.plot(
                pts_finite[order, 0], pts_finite[order, 1],
                linestyle="--", color=color, alpha=0.6, zorder=0,
            )

    n_excluded = int(np.sum(~np.all(np.isfinite(objectives), axis=1)))
    if n_excluded > 0:
        title = f"{title}\n({n_excluded} soluciones degeneradas excluidas, XB=inf)"

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small", title="Frentes")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    saved_path = None
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_path = out_path / filename
        fig.savefig(saved_path, dpi=150)
        print(f"[save] Gráfico de frentes de Pareto guardado en: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(saved_path) if saved_path is not None else None

def jaccard_index_solutions(Solutions_Matrix: np.ndarray) -> np.ndarray:
    xp = cp if _GPU else np   # mismo patrón que solution_encoding.py

    labels = xp.asarray(Solutions_Matrix)
    n_solutions, n_genes = labels.shape
    K = int(labels.max())     # etiquetas 1..K (build_clusters_population es 1-based)

    # One-hot por solución: (n_solutions, n_genes, K) — reemplaza los
    # vectores de pares O(n²) por una representación O(n·K).
    one_hot = xp.zeros((n_solutions, n_genes, K), dtype=xp.float32)
    idx_sol = xp.arange(n_solutions)[:, None]
    idx_gene = xp.arange(n_genes)[None, :]
    one_hot[idx_sol, idx_gene, labels - 1] = 1.0

    cluster_sizes = one_hot.sum(axis=1)                                   # (n_solutions, K)
    same_pairs = (cluster_sizes * (cluster_sizes - 1) / 2).sum(axis=1)    # (n_solutions,) == sum(v_i)

    J = xp.eye(n_solutions, dtype=xp.float64)
    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            N = one_hot[i].T @ one_hot[j]              # tabla de contingencia (K, K)
            r = float((N * (N - 1) / 2).sum())
            denom = float(same_pairs[i] + same_pairs[j]) - r
            J[i, j] = J[j, i] = r / denom if denom > 0 else 0.0

    return cp.asnumpy(J) if _GPU else J

def jaccard_population(generations_labels: list[np.ndarray]) -> list[np.ndarray]:
    """
    Índice de Jaccard entre individuos (labels) de cada generación, vía
    jaccard_index_solutions de biocluster. Boxplot con una caja por generación.
    """
    n_generations = len(generations_labels)
    jaccard_per_generation = []

    for labels_pop in generations_labels:
        J = jaccard_index_solutions(Solutions_Matrix=labels_pop)
        pop_size = J.shape[0]
        iu = np.triu_indices(pop_size, k=1)
        jaccard_per_generation.append(J[iu])

    plt.figure(figsize=(max(6, n_generations * 0.6), 6))
    plt.boxplot(jaccard_per_generation, vert=True, positions=range(1, n_generations + 1))
    plt.xlabel("Generación")
    plt.ylabel("Índice de Jaccard")
    plt.title(f"Similitud entre individuos por generación (n={n_generations})")
    plt.grid(True, alpha=0.3)
    plt.show()

    return jaccard_per_generation

def jaccard_f0_history(history_objectives: list[np.ndarray], history_labels: list[np.ndarray]) -> np.ndarray:
    """
    Jaccard promedio entre pares de soluciones de F0, generación a generación
    (misma definición que el Jaccard final del frente). NaN si |F0| < 2.

    F0 suele repetirse entre generaciones consecutivas, por lo que se cachea el
    resultado según los labels de F0 para no recalcular Jaccard innecesariamente.
    """
    from pareto_sorting import non_dominated_sort           # Import local: evita acoplar módulos al cargar.

    cache: dict[bytes, float] = {}
    out = np.full(len(history_objectives), np.nan, dtype=np.float64)
    for g, (obj, lab) in enumerate(zip(history_objectives, history_labels)):
        if len(obj) == 0:
            continue
        f0 = np.asarray(non_dominated_sort(obj)[0])
        if len(f0) < 2:
            continue
        labels_f0 = np.ascontiguousarray(np.asarray(lab)[f0])
        key = labels_f0.tobytes()                           # Mismo conjunto en el mismo orden => mismo valor.
        if key not in cache:
            J = jaccard_index_solutions(labels_f0)
            cache[key] = float(J[np.triu_indices(len(f0), k=1)].mean())
        out[g] = cache[key]
    return out

def _jaccard_mean_all_pairs(labels: np.ndarray) -> float:
    """
    Jaccard promedio sobre TODOS los pares de soluciones, calculado de una vez
    (equivale a jaccard_index_solutions + media del triángulo superior, pero con
    un único producto de matrices en vez de un bucle por pares).
    """
    xp = cp if _GPU else np
    lab = xp.asarray(labels)
    m, n = lab.shape
    K = int(lab.max())
    one_hot = xp.zeros((m, n, K), dtype=xp.float32)
    one_hot[xp.arange(m)[:, None], xp.arange(n)[None, :], lab - 1] = 1.0
    A = one_hot.transpose(1, 0, 2).reshape(n, m * K)
    G = (A.T @ A).astype(xp.float64).reshape(m, K, m, K)        # Tablas de contingencia de todos los pares.
    r = (G * (G - 1) / 2).sum(axis=(1, 3))                      # Pares en el mismo cluster en ambas soluciones.
    s = xp.diag(r)                                              # Pares en el mismo cluster dentro de cada solución.
    den = s[:, None] + s[None, :] - r
    J = xp.where(den > 0, r / xp.where(den > 0, den, 1.0), 0.0)
    iu = np.triu_indices(m, k=1)
    J = cp.asnumpy(J) if _GPU else J
    return float(J[iu].mean())

def jaccard_population_history(history_labels: list[np.ndarray]) -> np.ndarray:
    """Jaccard promedio entre TODOS los individuos de la población, generación a generación (NaN si hay < 2)."""
    out = np.full(len(history_labels), np.nan, dtype=np.float64)
    for g, lab in enumerate(history_labels):
        if len(lab) >= 2:
            out[g] = _jaccard_mean_all_pairs(np.asarray(lab))
    return out

# Punto de referencia (editable) para el hipervolumen de F0. Debe ser peor
# (mayor) que las soluciones del frente en ambos objetivos (minimización).
HV_REFERENCE_POINT = (120.0, 120.0)

def hypervolume_f1(f1_objectives: np.ndarray, ref_point: tuple[float, float] = HV_REFERENCE_POINT) -> float:
    """
    Hipervolumen (área) dominado por el frente F0 respecto a `ref_point`
    (por defecto HV_REFERENCE_POINT = (5,5)). MAYOR es mejor.

    f1_objectives : np.ndarray shape (m, 2) — (XB_GE, XB_BI) de las soluciones de F0.
    Soluciones que no dominan al punto de referencia (o con XB=inf) no aportan área.
    """
    pts = np.asarray(f1_objectives, dtype=np.float64).reshape(-1, 2)
    ref_ge, ref_bi = ref_point
    pts = pts[(pts[:, 0] < ref_ge) & (pts[:, 1] < ref_bi)]
    if len(pts) == 0:
        return 0.0

    # Barrido por XB_GE creciente, acumulando el mejor (mínimo) XB_BI visto.
    pts = pts[np.argsort(pts[:, 0], kind="stable")]
    hv = 0.0
    best_bi = ref_bi
    for ge, bi in pts:
        if bi < best_bi:
            hv += (ref_ge - ge) * (best_bi - bi)
            best_bi = bi
    return float(hv)

def plot_hypervolume_convergence(
    hv_f1_per_generation: list[float],
    output_dir: str | None = None,
    filename: str = "hypervolume_convergence.png",
    show: bool = False,
) -> str | None:
    """Convergencia del hipervolumen de F0 por generación (valor y mejor acumulado)."""
    hv = np.asarray(hv_f1_per_generation, dtype=np.float64)
    generations = np.arange(1, len(hv) + 1)
    best_so_far = np.maximum.accumulate(hv)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(generations, hv, color="steelblue", marker="o", label="HV de F0 por generación")
    ax.plot(generations, best_so_far, color="darkorange", linestyle="--", marker="s", label="Mejor acumulado (máximo)")

    ax.set_xlabel("Generación")
    ax.set_ylabel(f"Hipervolumen de F0, ref = {HV_REFERENCE_POINT} (mayor es mejor)")
    ax.set_title("Convergencia del hipervolumen de F0")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
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