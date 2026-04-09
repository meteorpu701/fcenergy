from pathlib import Path
import re
import numpy as np
import pandas as pd


BASE_DIR = Path("data")
OUT_DIR = BASE_DIR / "exp3_tables"
OUT_DIR.mkdir(exist_ok=True)

def parse_sigma(name: str) -> float:
    stem = Path(name).stem
    m = re.search(r"sigma_?([0-9]+(?:\.[0-9]+)?)", stem)
    if not m:
        raise ValueError(f"Could not parse sigma from {name}")
    return float(m.group(1))


def parse_sigma_seed(name: str):
    stem = Path(name).stem
    m = re.search(r"sigma_?([0-9]+(?:\.[0-9]+)?)_seed([0-9]+)", stem)
    if not m:
        raise ValueError(f"Could not parse sigma/seed from {name}")
    return float(m.group(1)), int(m.group(2))


def final_row(df: pd.DataFrame) -> pd.Series:
    if "round" in df.columns:
        return df.sort_values("round").iloc[-1]
    return df.iloc[-1]


def first_existing(row: pd.Series, candidates, default=np.nan):
    for c in candidates:
        if c in row.index:
            return row[c]
    return default


def summarise_clip_files(pattern: str, out_name: str, include_seed: bool = False):
    rows = []
    for path in sorted(BASE_DIR.glob(pattern)):
        df = pd.read_csv(path)
        last = final_row(df)

        sigma = parse_sigma(path.name)
        row = {
            "file": path.name,
            "sigma": sigma,
            "final_round": int(first_existing(last, ["round"], default=np.nan)),
            "rmse_ret": float(first_existing(last, ["rmse_ret"], default=np.nan)),
            "rmse_price_implied": float(first_existing(last, ["rmse_price_implied"], default=np.nan)),
            "baseline_rmse_ret": float(first_existing(last, ["baseline_rmse_ret"], default=np.nan)),
            "baseline_rmse_price_implied": float(first_existing(last, ["baseline_rmse_price_implied"], default=np.nan)),
        }

        if include_seed:
            _, seed = parse_sigma_seed(path.name)
            row["seed"] = seed

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["sigma"] + (["seed"] if include_seed else []))
    out.to_csv(OUT_DIR / out_name, index=False)
    return out


def summarise_canary_files():
    rows = []

    for path in sorted(BASE_DIR.glob("exp3_canary_sigma*_seed*.csv")):
        sigma, seed = parse_sigma_seed(path.name)
        df = pd.read_csv(path)
        last = final_row(df)

        row = {
            "file": path.name,
            "sigma": sigma,
            "seed": seed,
            "final_round": int(first_existing(last, ["round"], default=np.nan)),
            "rmse_ret": float(first_existing(last, ["rmse_ret"], default=np.nan)),
            "rmse_price_implied": float(first_existing(last, ["rmse_price_implied"], default=np.nan)),
            "canary_rmse": float(first_existing(last, ["canary_rmse"], default=np.nan)),
            "canary_frac": float(first_existing(last, ["canary_frac"], default=np.nan)),
            "canary_round_eval": float(first_existing(last, ["canary_round_eval"], default=np.nan)),
        }
        rows.append(row)

    by_seed = pd.DataFrame(rows).sort_values(["sigma", "seed"])
    by_seed.to_csv(OUT_DIR / "exp3_canary_final_by_seed.csv", index=False)

    agg = (
        by_seed.groupby("sigma", as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_final_round=("final_round", "mean"),
            std_final_round=("final_round", "std"),
            mean_rmse_ret=("rmse_ret", "mean"),
            std_rmse_ret=("rmse_ret", "std"),
            mean_rmse_price_implied=("rmse_price_implied", "mean"),
            std_rmse_price_implied=("rmse_price_implied", "std"),
            mean_canary_rmse=("canary_rmse", "mean"),
            std_canary_rmse=("canary_rmse", "std"),
            mean_canary_frac=("canary_frac", "mean"),
            std_canary_frac=("canary_frac", "std"),
        )
        .sort_values("sigma")
    )
    agg.to_csv(OUT_DIR / "exp3_canary_aggregated.csv", index=False)

    pretty = agg.copy()
    for col in pretty.columns:
        if col != "sigma" and col != "n_seeds":
            pretty[col] = pretty[col].round(4)
    pretty["sigma"] = pretty["sigma"].round(4)
    pretty.to_csv(OUT_DIR / "exp3_canary_aggregated_pretty.csv", index=False)

    thesis = pretty[
        [
            "sigma",
            "mean_rmse_ret",
            "std_rmse_ret",
            "mean_rmse_price_implied",
            "std_rmse_price_implied",
            "mean_canary_rmse",
            "std_canary_rmse",
        ]
    ].rename(
        columns={
            "sigma": "privacy_sigma",
            "mean_rmse_ret": "mean_final_rmse_ret",
            "std_rmse_ret": "std_final_rmse_ret",
            "mean_rmse_price_implied": "mean_final_rmse_price",
            "std_rmse_price_implied": "std_final_rmse_price",
            "mean_canary_rmse": "mean_final_canary_rmse",
            "std_canary_rmse": "std_final_canary_rmse",
        }
    )
    thesis.to_csv(OUT_DIR / "exp3_canary_table_for_thesis.csv", index=False)

    return by_seed, agg, thesis


def main():
    # Short clip-only setting
    clip_short = summarise_clip_files(
        pattern="exp3_clip_sigma_*.csv",
        out_name="exp3_clip_summary_short.csv",
        include_seed=False,
    )

    # Long clip-only setting
    clip_long = summarise_clip_files(
        pattern="exp3_long_clip_sigma_*_seed*.csv",
        out_name="exp3_clip_summary_long.csv",
        include_seed=True,
    )

    # Canary runs
    canary_by_seed, canary_agg, canary_thesis = summarise_canary_files()

    print("Wrote tables to:", OUT_DIR.resolve())
    print("\nFiles created:")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(" -", p.name)

    print("\nPreview: canary aggregated")
    print(canary_thesis.to_string(index=False))


if __name__ == "__main__":
    main()