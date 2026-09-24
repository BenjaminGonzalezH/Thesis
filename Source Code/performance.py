"""
Visualización y métricas de desempeño: frentes de Pareto, diversidad
(Jaccard) e hipervolumen respecto al origen.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import biocluster.clustering.jaccard_values as jv


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
        label = f"F{rank + 1}" if rank < max_fronts_highlighted else None

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


def jaccard_population(generations_labels: list[np.ndarray]) -> list[np.ndarray]:
    """
    Índice de Jaccard entre individuos (labels) de cada generación, vía
    jaccard_index_solutions de biocluster. Boxplot con una caja por generación.
    """
    n_generations = len(generations_labels)
    jaccard_per_generation = []

    for labels_pop in generations_labels:
        J = jv.jaccard_index_solutions(Solutions_Matrix=labels_pop)
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


def hypervolume_from_origin(xb_ge_pop: np.ndarray, xb_bi_pop: np.ndarray) -> np.ndarray:
    """
    Hipervolumen de cada solución respecto al origen (0,0): área
    XB_GE · XB_BI. El origen es el punto IDEAL, así que MENOR es mejor.
    """
    xb_ge_pop = np.asarray(xb_ge_pop, dtype=np.float64)
    xb_bi_pop = np.asarray(xb_bi_pop, dtype=np.float64)
    return xb_ge_pop * xb_bi_pop


def plot_hypervolume_convergence(
    hv_per_generation: list[np.ndarray],
    output_dir: str | None = None,
    filename: str = "hypervolume_convergence.png",
    show: bool = False,
) -> str | None:
    """Convergencia del hipervolumen por generación (promedio, rango y mejor acumulado)."""
    n_generations = len(hv_per_generation)
    generations = np.arange(1, n_generations + 1)

    mean_hv = np.array([np.mean(hv) for hv in hv_per_generation])
    min_hv = np.array([np.min(hv) for hv in hv_per_generation])
    max_hv = np.array([np.max(hv) for hv in hv_per_generation])
    best_so_far = np.minimum.accumulate(min_hv)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(generations, min_hv, max_hv, color="steelblue", alpha=0.2, label="Rango [mín, máx]")
    ax.plot(generations, mean_hv, color="steelblue", marker="o", label="Promedio por generación")
    ax.plot(generations, best_so_far, color="darkorange", linestyle="--", marker="s", label="Mejor acumulado (mínimo)")

    ax.set_xlabel("Generación")
    ax.set_ylabel("Hipervolumen (menor es mejor)")
    ax.set_title("Convergencia del hipervolumen")
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