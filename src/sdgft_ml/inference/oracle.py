"""Query interface for the SDGFT Oracle Database (Parquet).

The Oracle Database contains 61.7M+ parameter-space points pre-computed
through the GNN surrogate, with χ² scores against 21 experimental
measurements.

Usage::

    from sdgft_ml.inference import OracleDB

    db = OracleDB()                         # loads oracle_db.parquet
    top10 = db.best_fit(n=10)               # lowest χ² points
    gold = db.gold_standard()               # χ²/dof < 1.2 subset
    higgs = db.filter_observable("higgs_mass", 125.0, 125.5)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Default path
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_PARQUET = _PKG_ROOT / "data" / "oracle_db.parquet"
_DEFAULT_GOLD = _PKG_ROOT / "data" / "oracle_gold.parquet"


class OracleDB:
    """Query interface for the SDGFT Oracle Database.

    Parameters
    ----------
    parquet_path : str or Path, optional
        Path to ``oracle_db.parquet``. Defaults to ``<project>/data/oracle_db.parquet``.
    gold_path : str or Path, optional
        Path to ``oracle_gold.parquet`` (Gold Standard subset).
    lazy : bool
        If True (default), data is loaded only when first accessed.

    Attributes
    ----------
    df : pandas.DataFrame
        The full Oracle Database (loaded on first access).
    n_rows : int
        Number of rows in the database.
    columns : list[str]
        Column names.
    """

    def __init__(
        self,
        parquet_path: str | Path | None = None,
        gold_path: str | Path | None = None,
        lazy: bool = True,
    ):
        self.parquet_path = Path(parquet_path or _DEFAULT_PARQUET)
        self.gold_path = Path(gold_path or _DEFAULT_GOLD)
        self._df = None
        self._gold_df = None

        if not lazy:
            _ = self.df

    @property
    def df(self) -> "pd.DataFrame":
        """The full Oracle Database DataFrame (lazy-loaded)."""
        if self._df is None:
            import pandas as pd

            if not self.parquet_path.exists():
                raise FileNotFoundError(
                    f"Oracle DB not found: {self.parquet_path}\n"
                    f"Ensure oracle_db.parquet is in the data/ directory."
                )
            self._df = pd.read_parquet(self.parquet_path)
        return self._df

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def best_fit(self, n: int = 10) -> "pd.DataFrame":
        """Return the N points with lowest total χ².

        Parameters
        ----------
        n : int
            Number of best-fit points to return.

        Returns
        -------
        DataFrame sorted by total_chi2 ascending.
        """
        return self.df.nsmallest(n, "total_chi2")

    def gold_standard(self) -> "pd.DataFrame":
        """Return the Gold Standard subset (χ²/dof < 1.2).

        Uses the pre-filtered gold Parquet if available,
        otherwise filters from the main database.
        """
        if self._gold_df is not None:
            return self._gold_df

        if self.gold_path.exists():
            import pandas as pd
            self._gold_df = pd.read_parquet(self.gold_path)
        else:
            self._gold_df = self.df[self.df["gold_standard"] == True].copy()
        return self._gold_df

    def filter_observable(
        self,
        name: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> "pd.DataFrame":
        """Filter database by observable value range.

        Parameters
        ----------
        name : str
            Observable column name (e.g. "higgs_mass", "n_s").
        min_val, max_val : float, optional
            Range bounds (inclusive).

        Returns
        -------
        Filtered DataFrame.
        """
        if name not in self.df.columns:
            raise KeyError(
                f"Observable '{name}' not in database. "
                f"Available: {[c for c in self.columns if c not in ('delta', 'delta_g', 'total_chi2', 'chi2_per_dof', 'n_tensions', 'gold_standard', 'desi_w_match')]}"
            )
        mask = np.ones(len(self.df), dtype=bool)
        if min_val is not None:
            mask &= self.df[name].values >= min_val
        if max_val is not None:
            mask &= self.df[name].values <= max_val
        return self.df[mask]

    def query(self, expr: str) -> "pd.DataFrame":
        """Run a pandas query expression on the database.

        Parameters
        ----------
        expr : str
            Pandas query string, e.g. "higgs_mass > 125 and n_s < 0.97"

        Returns
        -------
        Filtered DataFrame.

        Example
        -------
        >>> db.query("higgs_mass > 125 and chi2_per_dof < 1.0")
        """
        return self.df.query(expr)

    def parameter_range(self) -> dict[str, dict[str, float]]:
        """Return min/max for each parameter in the database."""
        return {
            "delta": {
                "min": float(self.df["delta"].min()),
                "max": float(self.df["delta"].max()),
            },
            "delta_g": {
                "min": float(self.df["delta_g"].min()),
                "max": float(self.df["delta_g"].max()),
            },
        }

    def chi2_heatmap(
        self,
        bins: int = 500,
        delta_range: tuple[float, float] | None = None,
        delta_g_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a 2D binned minimum-χ² heatmap.

        Parameters
        ----------
        bins : int
            Number of bins per axis.
        delta_range, delta_g_range : tuple, optional
            Override axis ranges.

        Returns
        -------
        (chi2_grid, delta_edges, delta_g_edges) : tuple of arrays
            chi2_grid has shape (bins, bins), NaN where no data.
        """
        d = self.df["delta"].values
        dg = self.df["delta_g"].values
        chi2 = self.df["total_chi2"].values

        if delta_range is None:
            delta_range = (d.min(), d.max())
        if delta_g_range is None:
            delta_g_range = (dg.min(), dg.max())

        d_edges = np.linspace(delta_range[0], delta_range[1], bins + 1)
        dg_edges = np.linspace(delta_g_range[0], delta_g_range[1], bins + 1)

        i_d = np.clip(np.digitize(d, d_edges) - 1, 0, bins - 1).astype(np.int32)
        i_dg = np.clip(np.digitize(dg, dg_edges) - 1, 0, bins - 1).astype(np.int32)

        grid = np.full((bins, bins), np.nan, dtype=np.float32)
        # Vectorized binned minimum using pandas for speed
        import pandas as pd
        tmp = pd.DataFrame({"r": i_d, "c": i_dg, "chi2": chi2})
        mins = tmp.groupby(["r", "c"])["chi2"].min()
        for (r, c), val in mins.items():
            grid[r, c] = val

        return grid, d_edges, dg_edges

    def summary(self) -> str:
        """Return a summary string of the database."""
        pr = self.parameter_range()
        n_gold = int(self.df["gold_standard"].sum()) if "gold_standard" in self.df.columns else "?"
        return (
            f"SDGFT Oracle Database\n"
            f"  Rows:       {self.n_rows:,}\n"
            f"  Columns:    {len(self.columns)}\n"
            f"  Δ range:    [{pr['delta']['min']:.6f}, {pr['delta']['max']:.6f}]\n"
            f"  δ_g range:  [{pr['delta_g']['min']:.6f}, {pr['delta_g']['max']:.6f}]\n"
            f"  Gold Std:   {n_gold:,}\n"
            f"  Best χ²:    {self.df['total_chi2'].min():.4f}\n"
            f"  File:       {self.parquet_path}\n"
            f"  Size:       {self.parquet_path.stat().st_size / 1e9:.2f} GB"
        )

    def __repr__(self) -> str:
        status = f"{self.n_rows:,} rows" if self._df is not None else "not loaded"
        return f"OracleDB({status}, path='{self.parquet_path.name}')"

    def __len__(self) -> int:
        return self.n_rows
