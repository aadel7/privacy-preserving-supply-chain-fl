"""
Partition the DataCo dataset into isolated client silos.

Two strategies are written side-by-side so experiments can mount the correct one:

  data/geographic/client_1/data.csv   # Europe
  data/geographic/client_2/data.csv   # LATAM
  data/shipping_mode/client_1/data.csv  # Express (First Class + Same Day)
  data/shipping_mode/client_2/data.csv  # Standard (Second + Standard Class)

For convenience, data/client_1 and data/client_2 are also refreshed as a copy of
the shipping-mode partition (used by the latest root docker-compose stack).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def find_raw_csv() -> Path:
    preferred = DATA_DIR / "DataCoSupplyChainDataset.csv"
    if preferred.is_file():
        return preferred

    csv_files = [
        p
        for p in DATA_DIR.glob("*.csv")
        if p.name not in {"data.csv"}
        and "client" not in p.name.lower()
    ]
    # Prefer the largest CSV (raw dataset)
    if csv_files:
        return max(csv_files, key=lambda p: p.stat().st_size)

    raise FileNotFoundError(
        f"Could not find the raw DataCo dataset CSV in {DATA_DIR}"
    )


def save_partition(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(df)} rows)")


def partition_geographic(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Market" not in df.columns:
        raise KeyError("Column 'Market' not found — needed for geographic partition")

    markets = df["Market"].dropna().unique().tolist()
    print(f"Markets found: {markets}")

    europe = df[df["Market"] == "Europe"].copy()
    latam = df[df["Market"] == "LATAM"].copy()

    if europe.empty or latam.empty:
        raise ValueError(
            f"Geographic split produced empty silo(s). Europe={len(europe)}, LATAM={len(latam)}. "
            f"Available markets: {markets}"
        )

    print(f"Client 1 (Europe): {len(europe)} rows")
    print(f"Client 2 (LATAM):  {len(latam)} rows")
    return europe, latam


def partition_shipping_mode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Shipping Mode" not in df.columns:
        raise KeyError("Column 'Shipping Mode' not found")

    print(f"Shipping modes found: {df['Shipping Mode'].unique().tolist()}")

    express_modes = ["First Class", "Same Day"]
    standard_modes = ["Second Class", "Standard Class"]

    express = df[df["Shipping Mode"].isin(express_modes)].copy()
    standard = df[df["Shipping Mode"].isin(standard_modes)].copy()

    print(f"Client 1 (Express Hub {express_modes}): {len(express)} rows")
    print(f"Client 2 (Standard Hub {standard_modes}): {len(standard)} rows")
    return express, standard


def copy_as_default_clients(src_c1: Path, src_c2: Path) -> None:
    """Keep data/client_1 and data/client_2 aligned with shipping-mode default."""
    dst_c1 = DATA_DIR / "client_1" / "data.csv"
    dst_c2 = DATA_DIR / "client_2" / "data.csv"
    dst_c1.parent.mkdir(parents=True, exist_ok=True)
    dst_c2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_c1, dst_c1)
    shutil.copy2(src_c2, dst_c2)
    print(f"  Default mounts refreshed: {dst_c1} and {dst_c2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition DataCo into FL client silos")
    parser.add_argument(
        "--mode",
        choices=["all", "geographic", "shipping_mode"],
        default="all",
        help="Which partition(s) to write (default: all)",
    )
    args = parser.parse_args()

    raw_path = find_raw_csv()
    print(f"Loading raw dataset: {raw_path}")
    df = pd.read_csv(raw_path, encoding="latin1")
    print(f"Loaded {len(df)} rows")

    if args.mode in ("all", "geographic"):
        print("\n=== Geographic partition (Europe vs LATAM) ===")
        c1, c2 = partition_geographic(df)
        p1 = DATA_DIR / "geographic" / "client_1" / "data.csv"
        p2 = DATA_DIR / "geographic" / "client_2" / "data.csv"
        save_partition(c1, p1)
        save_partition(c2, p2)

    if args.mode in ("all", "shipping_mode"):
        print("\n=== Shipping Mode partition (Express vs Standard) ===")
        c1, c2 = partition_shipping_mode(df)
        p1 = DATA_DIR / "shipping_mode" / "client_1" / "data.csv"
        p2 = DATA_DIR / "shipping_mode" / "client_2" / "data.csv"
        save_partition(c1, p1)
        save_partition(c2, p2)
        copy_as_default_clients(p1, p2)

    print("\nPartitioning complete.")
    print("Use:")
    print("  experiments/01_...  → data/geographic/client_*/data.csv")
    print("  experiments/02+     → data/shipping_mode/client_*/data.csv")


if __name__ == "__main__":
    main()
