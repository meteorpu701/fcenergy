from pathlib import Path
import re
import math
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("data")
OUT_DIR = BASE_DIR / "chapter5_plots"
OUT_DIR.mkdir(exist_ok=True)

EXP1_LOG = BASE_DIR / "exp1_fedavg_abides_log.csv"
EXP2B = BASE_DIR / "exp2b_grid_summary.csv"
EXP2C = BASE_DIR / "exp2c_grid_summary_long.csv"

EXP3_FILES = sorted(BASE_DIR.glob("exp3_canary_sigma*_seed*.csv"))

def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def extract_sigma_seed(path: Path):
    m = re.search(r"sigma([0-9.]+)_seed([0-9]+)", path.name)
    if not m:
        raise ValueError(f"Could not parse sigma/seed from {path.name}")
    return float(m.group(1)), int(m.group(2))


def final_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("round").iloc[-1]


def style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = ""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_exp1():
    df = pd.read_csv(EXP1_LOG)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["round"], df["test_rmse"], marker="o", markersize=3)
    best_idx = df["test_rmse"].idxmin()
    best_round = int(df.loc[best_idx, "round"])
    best_rmse = float(df.loc[best_idx, "test_rmse"])
    ax.scatter([best_round], [best_rmse], s=50)
    ax.annotate(
        f"Best round = {best_round}\nRMSE = {best_rmse:.4f}",
        xy=(best_round, best_rmse),
        xytext=(10, 10),
        textcoords="offset points",
    )
    style_axes(
        ax,
        xlabel="Communication round",
        ylabel="Test RMSE",
        title="Exp1: FedAvg test RMSE across rounds",
    )
    savefig("exp1_fedavg_test_rmse_rounds.png")

    methods = [
        "Naive AR(1)",
        "Centralized AR-only Ridge",
        "Centralized ARX Ridge",
        "FedAvg (best round)",
        "FedAvg (final round)",
    ]
    values = [
        1.0092997572574753,
        1.508542867768617,
        1.8795357785930287,
        best_rmse,
        float(df.iloc[-1]["test_rmse"]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(methods, values)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    style_axes(
        ax,
        ylabel="RMSE",
        title="Exp1: baseline comparison",
    )
    savefig("exp1_baseline_comparison.png")

def plot_grouped_bars(df: pd.DataFrame, title: str, output_name: str):
    pivot = df.pivot(index="test_hub", columns="algo", values="rmse_ret_mean")

    hubs = list(pivot.index)
    algos = list(pivot.columns)

    x = range(len(hubs))
    n = len(algos)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, algo in enumerate(algos):
        offsets = [xi - 0.4 + width / 2 + i * width for xi in x]
        ax.bar(offsets, pivot[algo].values, width=width, label=algo)

    ax.set_xticks(list(x))
    ax.set_xticklabels(hubs)
    ax.legend(title="Method")
    style_axes(
        ax,
        xlabel="Held-out hub",
        ylabel="Mean RMSE (return space)",
        title=title,
    )
    savefig(output_name)


def plot_overall_avg(df: pd.DataFrame, title: str, output_name: str):
    avg = (
        df.groupby("algo", as_index=False)["rmse_ret_mean"]
        .mean()
        .sort_values("rmse_ret_mean")
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(avg["algo"], avg["rmse_ret_mean"])
    ax.set_xticklabels(avg["algo"], rotation=20, ha="right")
    style_axes(
        ax,
        ylabel="Average RMSE across hubs",
        title=title,
    )
    savefig(output_name)


def plot_exp2_comparison(exp2b: pd.DataFrame, exp2c: pd.DataFrame):
    avg_b = exp2b.groupby("algo", as_index=False)["rmse_ret_mean"].mean()
    avg_b["setting"] = "Exp2b"

    avg_c = exp2c.groupby("algo", as_index=False)["rmse_ret_mean"].mean()
    avg_c["setting"] = "Exp2c"

    combined = pd.concat([avg_b, avg_c], ignore_index=True)

    methods = sorted(combined["algo"].unique())
    settings = ["Exp2b", "Exp2c"]

    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, setting in enumerate(settings):
        sub = (
            combined[combined["setting"] == setting]
            .set_index("algo")
            .reindex(methods)
            .reset_index()
        )
        offsets = [xi - width / 2 + i * width for xi in x]
        ax.bar(offsets, sub["rmse_ret_mean"], width=width, label=setting)

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.legend()
    style_axes(
        ax,
        ylabel="Average RMSE across hubs",
        title="Exp2b vs Exp2c: overall aggregation-method comparison",
    )
    savefig("exp2b_vs_exp2c_overall_comparison.png")


def plot_exp2():
    exp2b = pd.read_csv(EXP2B)
    exp2c = pd.read_csv(EXP2C)

    plot_grouped_bars(
        exp2b,
        title="Exp2b: mean return RMSE by held-out hub and method",
        output_name="exp2b_grouped_by_hub.png",
    )
    plot_overall_avg(
        exp2b,
        title="Exp2b: overall average RMSE across held-out hubs",
        output_name="exp2b_overall_average.png",
    )

    plot_grouped_bars(
        exp2c,
        title="Exp2c: mean return RMSE by held-out hub and method",
        output_name="exp2c_grouped_by_hub.png",
    )
    plot_overall_avg(
        exp2c,
        title="Exp2c: overall average RMSE across held-out hubs",
        output_name="exp2c_overall_average.png",
    )

    plot_exp2_comparison(exp2b, exp2c)

def load_exp3_final_summary():
    rows = []
    for path in EXP3_FILES:
        sigma, seed = extract_sigma_seed(path)
        df = pd.read_csv(path)
        last = final_row(df)

        rows.append(
            {
                "file": path.name,
                "sigma": sigma,
                "seed": seed,
                "final_round": int(last["round"]),
                "rmse_ret": float(last["rmse_ret"]),
                "rmse_price_implied": float(last["rmse_price_implied"]),
                "canary_rmse": float(last["canary_rmse"]) if pd.notna(last["canary_rmse"]) else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["sigma", "seed"])


def plot_exp3():
    summary = load_exp3_final_summary()

    agg = (
        summary.groupby("sigma", as_index=False)
        .agg(
            rmse_ret_mean=("rmse_ret", "mean"),
            rmse_ret_std=("rmse_ret", "std"),
            rmse_price_mean=("rmse_price_implied", "mean"),
            rmse_price_std=("rmse_price_implied", "std"),
            canary_mean=("canary_rmse", "mean"),
            canary_std=("canary_rmse", "std"),
        )
        .sort_values("sigma")
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        agg["sigma"],
        agg["canary_mean"],
        yerr=agg["canary_std"],
        marker="o",
        capsize=4,
    )
    style_axes(
        ax,
        xlabel="Privacy noise σ",
        ylabel="Final canary RMSE",
        title="Exp3: canary RMSE vs privacy noise",
    )
    savefig("exp3_canary_rmse_vs_sigma.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        agg["sigma"],
        agg["rmse_ret_mean"],
        yerr=agg["rmse_ret_std"],
        marker="o",
        capsize=4,
    )
    style_axes(
        ax,
        xlabel="Privacy noise σ",
        ylabel="Final RMSE (return space)",
        title="Exp3: forecast utility vs privacy noise",
    )
    savefig("exp3_rmse_ret_vs_sigma.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        agg["sigma"],
        agg["rmse_price_mean"],
        yerr=agg["rmse_price_std"],
        marker="o",
        capsize=4,
    )
    style_axes(
        ax,
        xlabel="Privacy noise σ",
        ylabel="Final price-implied RMSE",
        title="Exp3: price-implied RMSE vs privacy noise",
    )
    savefig("exp3_rmse_price_vs_sigma.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(agg["canary_mean"], agg["rmse_ret_mean"])
    for _, row in agg.iterrows():
        ax.annotate(f"σ={row['sigma']:g}", (row["canary_mean"], row["rmse_ret_mean"]), xytext=(5, 5), textcoords="offset points")
    style_axes(
        ax,
        xlabel="Mean final canary RMSE",
        ylabel="Mean final RMSE (return space)",
        title="Exp3: privacy–utility trade-off",
    )
    savefig("exp3_privacy_utility_scatter.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for seed, sub in summary.groupby("seed"):
        sub = sub.sort_values("sigma")
        ax.plot(sub["sigma"], sub["canary_rmse"], marker="o", label=f"seed={seed}")
    ax.legend()
    style_axes(
        ax,
        xlabel="Privacy noise σ",
        ylabel="Final canary RMSE",
        title="Exp3: per-seed canary RMSE across privacy noise",
    )
    savefig("exp3_canary_per_seed.png")

    summary.to_csv(OUT_DIR / "exp3_final_round_per_seed_summary.csv", index=False)
    agg.to_csv(OUT_DIR / "exp3_final_round_aggregated_summary.csv", index=False)


def main():
    plot_exp1()
    plot_exp2()
    plot_exp3()
    print(f"Saved plots to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()