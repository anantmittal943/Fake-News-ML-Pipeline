from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_DATA_PATH

SEPARATOR = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def run_profile(data_path: Path = DEFAULT_DATA_PATH) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place 32_fake_news.csv in project root or pass --data-path."
        )

    df_raw = pd.read_csv(data_path, dtype=str)
    df = pd.read_csv(data_path)

    section("SCHEMA AUDIT")
    print(df.dtypes)
    print(f"\nShape: {df.shape}")

    section("TYPE-MISMATCH CHECK")
    for col in df_raw.columns:
        non_numeric = df_raw[col][pd.to_numeric(df_raw[col], errors="coerce").isna()]
        if len(non_numeric) > 0:
            print(
                f"  [WARN] {col}: {len(non_numeric)} un-parseable value(s) -> {non_numeric.unique()[:5]}"
            )
        else:
            print(f"  [OK]   {col}: all values parse as numbers")

    section("EXPLICIT MISSING VALUES")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  No explicit NaNs found")

    section("DOMAIN VIOLATIONS")
    checks = {
        "caps_ratio > 1.0": df["caps_ratio"] > 1.0,
        "caps_ratio < 0.0": df["caps_ratio"] < 0.0,
        "word_count <= 0": df["word_count"] <= 0,
        "exclamation_count < 0": df["exclamation_count"] < 0,
        "source_credibility_rank outside [1,100]": ~df["source_credibility_rank"].between(1, 100),
    }

    for desc, mask in checks.items():
        count = int(mask.sum())
        mark = "[WARN]" if count > 0 else "[OK]"
        print(f"  {mark} {desc}: {count} rows")

    section("SENTINEL CHECK")
    sentinels = [0, -1, -999, 999, 9999]
    for col in ["word_count", "caps_ratio", "exclamation_count", "source_credibility_rank"]:
        for sentinel in sentinels:
            n = int((df[col] == sentinel).sum())
            if n:
                pct = 100 * n / len(df)
                print(f"  [WARN] {col} == {sentinel}: {n} rows ({pct:.1f}%)")

    section("DEEP DIVE: caps_ratio == 0")
    zero_mask = df["caps_ratio"] == 0.0
    n_zeros = int(zero_mask.sum())
    pct_zeros = 100 * n_zeros / len(df)

    print(f"Rows with caps_ratio == 0: {n_zeros} ({pct_zeros:.1f}%)")
    print(f"All have word_count > 0: {(df.loc[zero_mask, 'word_count'] > 0).all()}")

    by_class = df.groupby("fake")["caps_ratio"].apply(lambda x: (x == 0).mean())
    print("Fraction of each class with caps_ratio == 0:")
    print(by_class.rename({0: "Real (fake=0)", 1: "Fake (fake=1)"}))


def main() -> None:
    run_profile()


if __name__ == "__main__":
    main()
