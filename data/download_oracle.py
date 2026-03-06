#!/usr/bin/env python3
"""Download the SDGFT Oracle Database from Zenodo.

Two Parquet files (~5.1 GB total):
  oracle_db.parquet    — 61.7 M parameter points, 44 columns  (~3.2 GB)
  oracle_gold.parquet  — 35 M Gold-Standard points            (~1.9 GB)

Published as:
  Besemer, D.A. (2026). SDGFT Oracle Database.
  Zenodo. https://doi.org/10.5281/zenodo.18863347

Usage:
    python download_oracle.py               # downloads to data/
    python download_oracle.py --dest /tmp   # custom destination
    python download_oracle.py --no-verify   # skip size verification
    python download_oracle.py --retries 5   # more retries on flaky networks

The script is idempotent — existing files whose size matches the
expected value are skipped without re-downloading.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ── File catalogue ────────────────────────────────────────────────────────────

ZENODO_BASE = "https://zenodo.org/records/18863347/files"

FILES: dict[str, dict] = {
    "oracle_db.parquet": {
        "url": f"{ZENODO_BASE}/oracle_db.parquet",
        "size_gb": 3.2,
        "description": "Full Oracle DB — 61.7 M parameter points, 44 columns",
    },
    "oracle_gold.parquet": {
        "url": f"{ZENODO_BASE}/oracle_gold.parquet",
        "size_gb": 1.9,
        "description": "Gold Standard subset — 35 M points, χ²/dof < 1.2",
    },
}

HEADERS = {
    "User-Agent": (
        "SDGFT-ML-Toolkit/1.0 "
        "(https://github.com/cosmologicmind/sdgft-ml-toolkit)"
    )
}

MANUAL_INSTRUCTIONS = """
─────────────────────────────────────────────────────────────────
Download failed. Manual instructions:

  Visit https://doi.org/10.5281/zenodo.18863347 and download:
    • oracle_db.parquet    (~3.2 GB)
    • oracle_gold.parquet  (~1.9 GB)

  Place both files in the  data/  directory.
─────────────────────────────────────────────────────────────────
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


def _progress_bar(received: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"  {_format_size(received)}"
    frac = min(received / total, 1.0)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    return f"  [{bar}] {pct:5.1f}%  {_format_size(received)} / {_format_size(total)}"


def _download_with_progress(
    url: str,
    dest: Path,
    *,
    timeout: int,
    retries: int,
    chunk: int = 1 << 20,  # 1 MB chunks
) -> None:
    """Download *url* → *dest* with a live progress bar, retrying on errors."""
    req = Request(url, headers=HEADERS)
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                received = 0
                with dest.open("wb") as fh:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
                        received += len(buf)
                        print(
                            f"\r{_progress_bar(received, total)}",
                            end="",
                            flush=True,
                        )
            print()  # newline after progress bar
            return
        except (URLError, OSError) as exc:
            last_exc = exc
            dest.unlink(missing_ok=True)
            if attempt < retries:
                wait = 2 ** (attempt - 1)
                print(f"\n     ↺ {exc}  — retrying in {wait}s")
                time.sleep(wait)

    print()
    raise last_exc  # type: ignore[misc]


# ── Main ──────────────────────────────────────────────────────────────────────

def download_oracle(
    dest_dir: str | Path = "data",
    *,
    verify: bool = True,
    retries: int = 3,
    timeout: int = 60,
) -> Path:
    """Download all Oracle Database files to *dest_dir*.

    Parameters
    ----------
    dest_dir:
        Target directory (created if absent). Defaults to ``data/``.
    verify:
        Check file size after download; warn if it deviates more than 5%.
    retries:
        Attempts before giving up on a file.
    timeout:
        Network timeout in seconds (per read chunk, not total).

    Returns
    -------
    Resolved destination path.
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Oracle DB destination: {dest}\n")

    failures: list[str] = []

    for fname, meta in FILES.items():
        target = dest / fname
        expected_bytes = int(meta["size_gb"] * 1024 ** 3)

        # ── Already exists? ──────────────────────────────────────────────────
        if target.exists():
            actual = target.stat().st_size
            if not verify or abs(actual - expected_bytes) / expected_bytes < 0.05:
                print(
                    f"  ✓ {fname} (cached, {_format_size(actual)})"
                )
                continue
            else:
                print(
                    f"  ⚠ {fname} exists but size mismatch "
                    f"({_format_size(actual)} vs ~{meta['size_gb']:.1f} GB) — re-downloading"
                )
                target.unlink()

        # ── Download ─────────────────────────────────────────────────────────
        print(f"  ↓ {fname}  ({meta['size_gb']:.1f} GB)")
        print(f"    {meta['description']}")
        print(f"    {meta['url']}")
        try:
            _download_with_progress(
                meta["url"], target, timeout=timeout, retries=retries
            )
            actual = target.stat().st_size
            print(f"  ✓ {fname} ({_format_size(actual)})")
            if verify and abs(actual - expected_bytes) / expected_bytes > 0.05:
                print(
                    f"  ⚠ Size differs from expected ~{meta['size_gb']:.1f} GB — "
                    "file may be incomplete or the database was updated on Zenodo."
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {fname}: {exc}", file=sys.stderr)
            failures.append(fname)

    # ── Summary ───────────────────────────────────────────────────────────────
    if failures:
        print(f"\n✗ Failed: {', '.join(failures)}", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS, file=sys.stderr)
        sys.exit(1)

    total_gb = sum(
        (dest / f).stat().st_size for f in FILES if (dest / f).exists()
    ) / 1024 ** 3
    print(f"\n✓ Done — {len(FILES)} files, {total_gb:.1f} GB in {dest}")
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SDGFT Oracle Database from Zenodo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Destination directory (default: same folder as this script, i.e. data/)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip file-size verification",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Attempts per file before giving up",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds",
    )
    args = parser.parse_args()
    download_oracle(
        args.dest,
        verify=not args.no_verify,
        retries=args.retries,
        timeout=args.timeout,
    )
