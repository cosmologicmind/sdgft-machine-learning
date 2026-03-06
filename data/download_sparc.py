#!/usr/bin/env python3
"""Download SPARC galaxy database (Lelli, McGaugh & Schombert 2016, AJ 152, 157).

Tries multiple mirrors in order:
  1. astroweb.cwru.edu (HTTPS, original host)
  2. astroweb.cwru.edu (HTTP fallback)
  3. CDS/VizieR, Strasbourg (permanent archive)

Usage:
    python download_sparc.py              # downloads to data/sparc/
    python download_sparc.py --dest /tmp/sparc
    python download_sparc.py --retries 5  # more retries per mirror

The script is idempotent — existing files are skipped.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ── Mirror definitions ────────────────────────────────────────────────────────

# Each entry is (label, base_url, mrt_suffix, zip_suffix)
# CDS serves the files as table1.dat/table2.dat — we rename them locally.
MIRRORS = [
    (
        "astroweb HTTPS",
        "https://astroweb.cwru.edu/SPARC",
        {"SPARC_Lelli2016c.mrt": "SPARC_Lelli2016c.mrt",
         "MassModels_Lelli2016c.mrt": "MassModels_Lelli2016c.mrt"},
        "Rotmod_LTG.zip",
    ),
    (
        "astroweb HTTP",
        "http://astroweb.cwru.edu/SPARC",
        {"SPARC_Lelli2016c.mrt": "SPARC_Lelli2016c.mrt",
         "MassModels_Lelli2016c.mrt": "MassModels_Lelli2016c.mrt"},
        "Rotmod_LTG.zip",
    ),
    (
        "CDS/VizieR",
        "https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157",
        {"SPARC_Lelli2016c.mrt": "SPARC_Lelli2016c.mrt",
         "MassModels_Lelli2016c.mrt": "MassModels_Lelli2016c.mrt"},
        "Rotmod_LTG.zip",
    ),
]

HEADERS = {"User-Agent": "SDGFT-ML-Toolkit/1.0 (https://github.com/cosmologicmind/sdgft-ml-toolkit)"}

MANUAL_INSTRUCTIONS = """
─────────────────────────────────────────────────────────────────
All mirrors failed. Manual download instructions:

  1. Visit  http://astroweb.cwru.edu/SPARC/
     or     https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157/

  2. Download these files into  data/sparc/ :
       SPARC_Lelli2016c.mrt
       MassModels_Lelli2016c.mrt
       Rotmod_LTG.zip  (then unzip into data/sparc/)
─────────────────────────────────────────────────────────────────
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch(url: str, *, timeout: int, retries: int) -> bytes:
    """GET *url*, retry on transient errors with exponential back-off."""
    req = Request(url, headers=HEADERS)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  ↓ {url}" + (f" (attempt {attempt}/{retries})" if attempt > 1 else ""))
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except URLError as exc:
            last_exc = exc
            if attempt < retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s …
                print(f"     ↺ {exc}  — retrying in {wait}s")
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _try_file(base: str, remote_name: str, dest: Path, *,
              timeout: int, retries: int) -> bool:
    """Try to download one file; return True on success."""
    url = f"{base}/{remote_name}"
    try:
        data = _fetch(url, timeout=timeout, retries=retries)
        dest.write_bytes(data)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"     ✗ {exc}")
        return False


def _try_zip(base: str, remote_name: str, extract_to: Path, *,
             timeout: int, retries: int) -> bool:
    """Try to download and unzip the rotation-curve archive; return True on success."""
    url = f"{base}/{remote_name}"
    try:
        data = _fetch(url, timeout=timeout, retries=retries)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(extract_to)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"     ✗ {exc}")
        return False


# ── Main download logic ───────────────────────────────────────────────────────

def download_sparc(
    dest_dir: str | Path = "data/sparc",
    *,
    retries: int = 3,
    timeout_short: int = 60,
    timeout_zip: int = 180,
) -> Path:
    """Download all SPARC files to *dest_dir*, trying mirrors in order.

    Parameters
    ----------
    dest_dir:
        Target directory (created if absent).
    retries:
        Attempts per mirror before falling back to the next one.
    timeout_short / timeout_zip:
        Network timeouts [s] for plain files and the ZIP archive.

    Returns
    -------
    Resolved destination path.
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"SPARC destination: {dest}")

    # Track which items still need downloading
    mrt_done: dict[str, bool] = {}
    zip_done = False
    rotmod_dir = dest / "Rotmod_LTG"

    for label, base, name_map, zip_name in MIRRORS:
        pending_mrts = [k for k, ok in mrt_done.items() if not ok]
        need_mrt = [k for k in name_map if k not in mrt_done]
        need_zip = (not zip_done) and not (
            rotmod_dir.exists() and any(rotmod_dir.iterdir())
        )

        if not need_mrt and not pending_mrts and not need_zip:
            break  # everything already satisfied

        print(f"\n[{label}]")

        # ── MRT files ──
        for local_name, remote_name in name_map.items():
            if mrt_done.get(local_name):
                continue
            target = dest / local_name
            if target.exists():
                print(f"  ✓ {local_name} (cached)")
                mrt_done[local_name] = True
                continue
            ok = _try_file(base, remote_name, target,
                           timeout=timeout_short, retries=retries)
            mrt_done[local_name] = ok
            if ok:
                print(f"  ✓ {local_name} ({target.stat().st_size / 1024:.0f} KB)")

        # ── Rotation-curve ZIP ──
        if need_zip:
            if rotmod_dir.exists() and any(rotmod_dir.iterdir()):
                n = len(list(rotmod_dir.glob("*_rotmod.dat")))
                print(f"  ✓ Rotmod_LTG/ (cached, {n} files)")
                zip_done = True
            else:
                zip_done = _try_zip(base, zip_name, dest,
                                    timeout=timeout_zip, retries=retries)
                if zip_done:
                    n = len(list(rotmod_dir.glob("*_rotmod.dat")))
                    print(f"  ✓ Rotmod_LTG/ ({n} rotation curves)")

        if all(mrt_done.values()) and zip_done:
            break  # all done, no need to try further mirrors

    # ── Final status report ───────────────────────────────────────────────────
    failures = [k for k, ok in mrt_done.items() if not ok]
    if not zip_done:
        failures.append("Rotmod_LTG.zip")

    if failures:
        print(f"\n✗ Could not download: {', '.join(failures)}", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS, file=sys.stderr)
        sys.exit(1)

    n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"\n✓ Done — {n_files} files in {dest}")
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SPARC galaxy database (Lelli et al. 2016)"
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(os.path.dirname(__file__), "sparc"),
        help="Destination directory (default: data/sparc/)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Attempts per mirror before trying the next one (default: 3)",
    )
    args = parser.parse_args()
    download_sparc(args.dest, retries=args.retries)
