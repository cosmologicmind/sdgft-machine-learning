"""Visualization utilities for SDGFT-ML.

Plotting functions for:
- Training history (loss curves)
- Parameter landscape heatmaps
- Canary test reports
- Active learning progression
- DAG structure visualization
"""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_training_history(
    history: Any,
    title: str = "Training History",
    save_path: str | None = None,
) -> Any:
    """Plot training and validation loss curves.

    Parameters
    ----------
    history : TrainHistory or InverterHistory
    title : str
    save_path : str | None
        If provided, save to file instead of showing.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = range(1, len(history.train_loss) + 1)

    ax.semilogy(epochs, history.train_loss, label="Train", alpha=0.8)
    ax.semilogy(epochs, history.val_loss, label="Val", alpha=0.8)
    ax.axvline(
        history.best_epoch + 1, color="red", linestyle="--",
        alpha=0.5, label=f"Best (ep {history.best_epoch + 1})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_parameter_landscape(
    df: Any,
    observable: str = "n_s",
    x_col: str = "param_delta",
    y_col: str = "param_delta_g",
    title: str | None = None,
    save_path: str | None = None,
    mark_sdgft_point: bool = True,
) -> Any:
    """Plot a 2D heatmap of an observable over the parameter landscape.

    Parameters
    ----------
    df : DataFrame
        Output of sweep_to_dataframe().
    observable : str
        Observable to plot.
    x_col, y_col : str
        Parameter columns for x and y axes.
    mark_sdgft_point : bool
        If True, mark the true SDGFT point (Δ=5/24, δ_g=1/24).
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    if title is None:
        title = f"{observable} over (Δ, δ_g) space"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = df[x_col].values
    y = df[y_col].values
    z = df[observable].values

    # Triangulated contour plot
    triang = tri.Triangulation(x, y)
    tcf = ax.tricontourf(triang, z, levels=50, cmap="viridis")
    fig.colorbar(tcf, ax=ax, label=observable)

    if mark_sdgft_point:
        ax.plot(5.0 / 24.0, 1.0 / 24.0, "r*", markersize=15,
                label="SDGFT axiom point", zorder=5)
        ax.legend()

    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$\delta_g$")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_canary_report(
    result: dict[str, Any],
    save_path: str | None = None,
) -> Any:
    """Visualize the canary test results as a bar chart.

    Parameters
    ----------
    result : dict
        Output of canary_test().
    """
    import matplotlib.pyplot as plt

    names = list(result["relative_errors"].keys())
    errors = [result["relative_errors"][n] for n in names]

    # Sort by error
    sorted_indices = np.argsort(errors)[::-1]
    names = [names[i] for i in sorted_indices]
    errors = [errors[i] for i in sorted_indices]

    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(names) * 0.3)))
    colors = ["red" if e > 0.05 else "green" for e in errors]
    ax.barh(range(len(names)), errors, color=colors, alpha=0.7)
    ax.axvline(0.05, color="orange", linestyle="--",
               label="5% tolerance", alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Relative Error")
    ax.set_title(
        f"Canary Test: {result['status']} "
        f"({result['n_passed']}/{result['n_total']} passed)"
    )
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dag(
    save_path: str | None = None,
) -> Any:
    """Visualize the SDGFT observable DAG.

    Requires networkx and matplotlib.
    """
    import matplotlib.pyplot as plt

    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for DAG visualization")

    from ..data.dag_builder import build_dag

    adj, names = build_dag()

    G = nx.DiGraph()
    for name in names:
        G.add_node(name)
    for name, deps in adj.items():
        for dep in deps:
            G.add_edge(dep, name)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Layout: topological sort layers
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color="lightblue",
        node_size=600,
        font_size=6,
        arrows=True,
        arrowsize=10,
        edge_color="gray",
        alpha=0.8,
    )
    ax.set_title("SDGFT Observable DAG")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_anomaly_map(
    result: Any,
    save_path: str | None = None,
) -> Any:
    """Plot anomaly detection results on the parameter landscape.

    Parameters
    ----------
    result : AnomalyResult
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: anomaly scores
    ax = axes[0]
    sc = ax.scatter(
        result.params[:, 0], result.params[:, 1],
        c=result.scores, cmap="YlOrRd", s=10, alpha=0.7,
    )
    fig.colorbar(sc, ax=ax, label="Anomaly Score")
    ax.axhline(result.threshold, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$\delta_g$")
    ax.set_title("Anomaly Scores")

    # Right: anomaly mask
    ax = axes[1]
    normal = ~result.anomaly_mask
    ax.scatter(
        result.params[normal, 0], result.params[normal, 1],
        c="blue", s=10, alpha=0.3, label="Normal",
    )
    ax.scatter(
        result.params[result.anomaly_mask, 0],
        result.params[result.anomaly_mask, 1],
        c="red", s=20, alpha=0.8, label="Anomalous",
    )
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$\delta_g$")
    ax.set_title(f"Anomalies ({result.anomaly_mask.sum()} detected)")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
