"""
Visualización y métricas de desempeño: frentes de Pareto, diversidad
(Jaccard) e hipervolumen de F1 respecto a un punto de referencia.
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

# Punto de referencia (editable) para el hipervolumen de F1. Debe ser peor
# (mayor) que las soluciones del frente en ambos objetivos (minimización).
HV_REFERENCE_POINT = (1.0, 1.0)

def hypervolume_f1(f1_objectives: np.ndarray, ref_point: tuple[float, float] = HV_REFERENCE_POINT) -> float:
    """
    Hipervolumen (área) dominado por el frente F1 respecto a `ref_point`
    (por defecto HV_REFERENCE_POINT = (5,5)). MAYOR es mejor.

    f1_objectives : np.ndarray shape (m, 2) — (XB_GE, XB_BI) de las soluciones de F1.
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
    """Convergencia del hipervolumen de F1 por generación (valor y mejor acumulado)."""
    hv = np.asarray(hv_f1_per_generation, dtype=np.float64)
    generations = np.arange(1, len(hv) + 1)
    best_so_far = np.maximum.accumulate(hv)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(generations, hv, color="steelblue", marker="o", label="HV de F1 por generación")
    ax.plot(generations, best_so_far, color="darkorange", linestyle="--", marker="s", label="Mejor acumulado (máximo)")

    ax.set_xlabel("Generación")
    ax.set_ylabel(f"Hipervolumen de F1, ref = {HV_REFERENCE_POINT} (mayor es mejor)")
    ax.set_title("Convergencia del hipervolumen de F1")
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