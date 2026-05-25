from __future__ import annotations

import argparse
import gc
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import order_stability_batch as osb


DATASET_ROOT = Path("/home/haeylee/main/dataset/Classical")
DEFAULT_OUTPUT_DIR = (
    osb.REPO_ROOT
    / "experiments/conditional_density/results/adbench_order_sensitivity_screening_2026_05_15"
)


DEFAULT_DATASETS = [
    "12_fault.npz",
    "43_WDBC.npz",
    "6_cardio.npz",
    "7_Cardiotocography.npz",
    "18_Ionosphere.npz",
    "20_letter.npz",
    "27_PageBlocks.npz",
    "28_pendigits.npz",
    "29_Pima.npz",
    "2_annthyroid.npz",
    "37_Stamps.npz",
    "38_thyroid.npz",
    "40_vowels.npz",
    "41_Waveform.npz",
    "4_breastw.npz",
    "47_yeast.npz",
    "46_WPBC.npz",
]


def dataset_name_from_path(path: Path) -> str:
    return re.sub(r"^\d+_", "", path.stem)


def load_npz_bundle(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"]).astype(int)
    dataset_name = dataset_name_from_path(path)
    counts = pd.Series(y).value_counts().sort_index()
    return {
        "dataset_name": dataset_name,
        "dataset_path": path,
        "X": X,
        "y": y,
        "feature_names": [f"feature_{idx:02d}" for idx in range(X.shape[1])],
        "feature_name_source": "generic_npz_columns",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "normal_count": int(counts.get(0, 0)),
        "anomaly_count": int(counts.get(1, 0)),
        "anomaly_ratio": float(counts.get(1, 0) / len(y)),
    }


def build_split(bundle: dict[str, Any], *, test_size: float, random_seed: int) -> dict[str, Any]:
    X = bundle["X"]
    y = bundle["y"]
    all_indices = np.arange(len(X))
    normal_mask = y == 0
    anomaly_mask = y == 1

    X_normal = X[normal_mask]
    X_anomaly = X[anomaly_mask]
    idx_normal = all_indices[normal_mask]
    idx_anomaly = all_indices[anomaly_mask]

    X_train_normal, X_test_normal, _, idx_test_normal = train_test_split(
        X_normal,
        idx_normal,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    X_test = np.vstack([X_test_normal, X_anomaly])
    y_test = np.concatenate(
        [
            np.zeros(len(X_test_normal), dtype=int),
            np.ones(len(X_anomaly), dtype=int),
        ]
    )
    test_indices = np.concatenate([idx_test_normal, idx_anomaly])
    shuffle_order = np.random.default_rng(random_seed).permutation(len(X_test))

    train_df = pd.DataFrame(X_train_normal, columns=bundle["feature_names"])
    test_df = pd.DataFrame(X_test[shuffle_order], columns=bundle["feature_names"])
    test_df["y_test_binary"] = y_test[shuffle_order]
    test_df["test_index"] = test_indices[shuffle_order]
    test_df["source_type"] = np.where(test_df["y_test_binary"] == 1, "anomaly", "normal")
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": bundle["dataset_name"],
                "dataset_path": str(bundle["dataset_path"]),
                "n_samples": bundle["n_samples"],
                "n_features": bundle["n_features"],
                "anomaly_ratio": bundle["anomaly_ratio"],
                "train_normal_count": len(train_df),
                "test_normal_count": int((test_df["y_test_binary"] == 0).sum()),
                "test_anomaly_count": int((test_df["y_test_binary"] == 1).sum()),
                "feature_name_source": bundle["feature_name_source"],
            }
        ]
    )
    return {"train_df": train_df, "test_df": test_df, "manifest_df": manifest}


def safe_corr_stats(train_df: pd.DataFrame) -> dict[str, float | int]:
    X = train_df.to_numpy(dtype=float)
    std = np.nanstd(X, axis=0)
    keep = std > 1e-12
    X = X[:, keep]
    if X.shape[1] < 2:
        return {
            "n_nonconstant_features": int(X.shape[1]),
            "mean_abs_corr": np.nan,
            "frac_abs_corr_gt_0_5": np.nan,
            "effective_rank_ratio": np.nan,
        }
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    tri = np.triu_indices_from(corr, k=1)
    abs_corr = np.abs(corr[tri])
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    if eig.sum() > 0:
        p = eig / eig.sum()
        p = p[p > 0]
        effective_rank = float(np.exp(-(p * np.log(p)).sum()))
        effective_rank_ratio = effective_rank / X.shape[1]
    else:
        effective_rank_ratio = np.nan
    return {
        "n_nonconstant_features": int(X.shape[1]),
        "mean_abs_corr": float(abs_corr.mean()),
        "frac_abs_corr_gt_0_5": float((abs_corr >= 0.5).mean()),
        "effective_rank_ratio": float(effective_rank_ratio),
    }


def select_orders(feature_names: list[str], *, n_random_orders: int, random_seed: int) -> list[dict[str, Any]]:
    return osb.build_order_bank(
        feature_names,
        n_random_orders=n_random_orders,
        random_seed=random_seed,
    )


def run_one_dataset(
    path: Path,
    *,
    output_dir: Path,
    device: str,
    n_estimators: int,
    random_seed: int,
    test_size: float,
    n_random_orders: int,
    max_tabpfn_train_samples: int | None,
    log_prob_floor: float | None,
) -> dict[str, pd.DataFrame]:
    osb.set_global_seed(random_seed)
    bundle = load_npz_bundle(path)
    split = build_split(bundle, test_size=test_size, random_seed=random_seed)
    train_df_full = split["train_df"]
    train_df = osb.sample_df(train_df_full, max_tabpfn_train_samples, random_seed)
    test_df = split["test_df"]
    orders = select_orders(bundle["feature_names"], n_random_orders=n_random_orders, random_seed=random_seed)

    sample_score_frames: list[pd.DataFrame] = []
    order_metrics: list[dict[str, Any]] = []
    timing_frames: list[pd.DataFrame] = []
    order_manifest: list[dict[str, Any]] = []

    for order in orders:
        order_name = order["order_name"]
        print(f"[run] dataset={bundle['dataset_name']} order={order_name}", flush=True)
        steps = osb.build_full_chain_scheme_from_order(order["feature_order"])
        eval_df, timing_df, metrics = osb.fit_and_score_order(
            order_name=order_name,
            order_type=order["order_type"],
            steps=steps,
            train_df=train_df,
            test_df=test_df,
            device=device,
            n_estimators=n_estimators,
            random_seed=random_seed,
            log_prob_floor=log_prob_floor,
        )
        sample_score_frames.append(
            osb.build_sample_score_frame(
                eval_df,
                dataset_name=bundle["dataset_name"],
                order_name=order_name,
                order_type=order["order_type"],
                score_col=f"{order_name}__total_log_score",
            )
        )
        order_metrics.append({"dataset_name": bundle["dataset_name"], **metrics})
        timing_frames.append(timing_df.assign(dataset_name=bundle["dataset_name"]))
        order_manifest.append(
            {
                "dataset_name": bundle["dataset_name"],
                "order_name": order_name,
                "order_type": order["order_type"],
                "n_steps": len(steps),
                "avg_parents": float(np.mean([len(parents) for _, parents in steps])),
                "max_parents": int(max([len(parents) for _, parents in steps])),
                "order_signature": osb.order_signature_from_steps(steps),
            }
        )
        gc.collect()

    sample_scores = pd.concat(sample_score_frames, ignore_index=True)
    sample_stability = osb.summarize_sample_stability(sample_scores)
    stability_summary = osb.summarize_dataset_stability(
        sample_stability,
        dataset_name=bundle["dataset_name"],
        n_orders_tested=len(orders),
    )
    rank_pairwise = osb.build_order_rank_correlation(sample_scores, dataset_name=bundle["dataset_name"])
    rank_summary = osb.summarize_rank_correlation(rank_pairwise, dataset_name=bundle["dataset_name"])

    manifest = split["manifest_df"].copy()
    manifest["n_orders_tested"] = len(orders)
    manifest["n_random_orders"] = n_random_orders
    manifest["train_normal_count_after_cap"] = len(train_df)
    manifest["max_tabpfn_train_samples"] = (
        max_tabpfn_train_samples if max_tabpfn_train_samples is not None else np.nan
    )
    manifest["device"] = device
    manifest["n_estimators"] = n_estimators
    manifest["log_prob_floor"] = log_prob_floor if log_prob_floor is not None else np.nan
    for key, value in safe_corr_stats(train_df_full).items():
        manifest[key] = value

    prefix = bundle["dataset_name"]
    manifest.to_csv(output_dir / f"{prefix}_dataset_manifest.csv", index=False)
    pd.DataFrame(order_manifest).to_csv(output_dir / f"{prefix}_order_manifest.csv", index=False)
    pd.DataFrame(order_metrics).to_csv(output_dir / f"{prefix}_order_metrics.csv", index=False)
    pd.concat(timing_frames, ignore_index=True).to_csv(output_dir / f"{prefix}_timing.csv", index=False)
    sample_scores.to_csv(output_dir / f"{prefix}_sample_scores.csv", index=False)
    sample_stability.to_csv(output_dir / f"{prefix}_sample_stability.csv", index=False)
    stability_summary.to_csv(output_dir / f"{prefix}_stability_summary.csv", index=False)
    rank_pairwise.to_csv(output_dir / f"{prefix}_rank_corr_pairwise.csv", index=False)
    rank_summary.to_csv(output_dir / f"{prefix}_rank_corr_summary.csv", index=False)

    return {
        "manifest": manifest,
        "order_metrics": pd.DataFrame(order_metrics),
        "timing": pd.concat(timing_frames, ignore_index=True),
        "sample_scores": sample_scores,
        "sample_stability": sample_stability,
        "stability_summary": stability_summary,
        "rank_corr_pairwise": rank_pairwise,
        "rank_corr_summary": rank_summary,
    }


def aggregate_and_plot(results: list[dict[str, pd.DataFrame]], output_dir: Path) -> None:
    combined: dict[str, pd.DataFrame] = {}
    key_to_file = {
        "manifest": "batch_dataset_manifest.csv",
        "order_metrics": "batch_order_metrics.csv",
        "timing": "batch_timing.csv",
        "sample_scores": "batch_sample_scores.csv",
        "sample_stability": "batch_sample_stability.csv",
        "stability_summary": "batch_stability_summary.csv",
        "rank_corr_pairwise": "batch_rank_corr_pairwise.csv",
        "rank_corr_summary": "batch_rank_corr_summary.csv",
    }
    for key, filename in key_to_file.items():
        frames = [result[key] for result in results if key in result and not result[key].empty]
        if frames:
            combined[key] = pd.concat(frames, ignore_index=True)
            combined[key].to_csv(output_dir / filename, index=False)

    manifest = combined["manifest"]
    stability = combined["stability_summary"]
    rank = combined["rank_corr_summary"]
    all_stab = stability[stability["source_group"] == "all"].copy()
    normal = stability[stability["source_group"] == "normal"][
        ["dataset_name", "mean_std_log_score"]
    ].rename(columns={"mean_std_log_score": "normal_mean_std"})
    anomaly = stability[stability["source_group"] == "anomaly"][
        ["dataset_name", "mean_std_log_score"]
    ].rename(columns={"mean_std_log_score": "anomaly_mean_std"})
    summary = (
        manifest.merge(all_stab, on="dataset_name", how="left")
        .merge(rank, on="dataset_name", how="left")
        .merge(normal, on="dataset_name", how="left")
        .merge(anomaly, on="dataset_name", how="left")
    )
    summary["ranking_instability"] = 1.0 - summary["mean_spearman_rho"]
    summary["anomaly_minus_normal_std"] = summary["anomaly_mean_std"] - summary["normal_mean_std"]
    summary["n_train_per_feature"] = summary["train_normal_count_after_cap"] / summary["n_features"]
    score_cols = ["mean_std_log_score", "p90_std_log_score", "ranking_instability"]
    for col in score_cols:
        summary[f"z_{col}"] = (summary[col] - summary[col].mean()) / summary[col].std(ddof=0)
    summary["screening_sensitivity_index"] = summary[[f"z_{col}" for col in score_cols]].mean(axis=1)
    summary = summary.sort_values("screening_sensitivity_index", ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "screening_summary_ranked.csv", index=False)

    colors = plt.cm.tab20(np.linspace(0, 1, len(summary)))
    ranked = summary.sort_values("mean_std_log_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.34 * len(ranked))))
    ax.barh(ranked["dataset_name"], ranked["mean_std_log_score"], color=colors[: len(ranked)])
    ax.set_xlabel("mean samplewise std across orders")
    ax.set_title("ADBench order sensitivity screening")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "screening_mean_std_ranked_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(summary["mean_std_log_score"], summary["ranking_instability"], s=65, alpha=0.82)
    for _, row in summary.iterrows():
        ax.annotate(row["dataset_name"], (row["mean_std_log_score"], row["ranking_instability"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean std_log_score")
    ax.set_ylabel("1 - mean Spearman rho")
    ax.set_title("Score instability vs ranking instability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "screening_score_vs_ranking_instability.png", dpi=180)
    plt.close(fig)

    scatter_specs = [
        ("n_features", "number of features"),
        ("n_train_per_feature", "normal train / feature"),
        ("frac_abs_corr_gt_0_5", "fraction |corr| >= 0.5"),
        ("effective_rank_ratio", "effective rank / d"),
        ("anomaly_ratio", "overall anomaly ratio"),
        ("anomaly_minus_normal_std", "anomaly - normal std"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for ax, (xcol, xlabel) in zip(axes.ravel(), scatter_specs):
        ax.scatter(summary[xcol], summary["mean_std_log_score"], s=60, alpha=0.82)
        for _, row in summary.iterrows():
            ax.annotate(row["dataset_name"], (row[xcol], row["mean_std_log_score"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean std_log_score")
        ax.grid(alpha=0.25)
    fig.suptitle("Dataset factors vs order sensitivity", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "screening_meta_feature_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_dataset_paths(names_or_files: list[str]) -> list[Path]:
    paths: list[Path] = []
    all_files = sorted(DATASET_ROOT.glob("*.npz"))
    by_name = {dataset_name_from_path(path): path for path in all_files}
    by_file = {path.name: path for path in all_files}
    for item in names_or_files:
        if item in by_file:
            paths.append(by_file[item])
        elif item in by_name:
            paths.append(by_name[item])
        else:
            path = Path(item)
            if path.exists():
                paths.append(path)
            else:
                raise ValueError(f"Unknown dataset: {item}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Broad ADBench order sensitivity screening.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--n-random-orders", type=int, default=3)
    parser.add_argument("--max-tabpfn-train-samples", type=int, default=1000)
    parser.add_argument("--log-prob-floor", type=str, default="1e-10")
    args = parser.parse_args()

    log_prob_floor = osb.parse_optional_float(args.log_prob_floor)
    device = "cuda" if args.device == "auto" and osb.torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_dataset_paths(args.datasets)
    print(f"[info] device={device}")
    print(f"[info] n_datasets={len(paths)} n_orders={args.n_random_orders + 2}")

    results: list[dict[str, pd.DataFrame]] = []
    for path in paths:
        started = time.time()
        print(f"[dataset] {dataset_name_from_path(path)} path={path.name}", flush=True)
        result = run_one_dataset(
            path,
            output_dir=args.output_dir,
            device=device,
            n_estimators=args.n_estimators,
            random_seed=args.random_seed,
            test_size=args.test_size,
            n_random_orders=args.n_random_orders,
            max_tabpfn_train_samples=args.max_tabpfn_train_samples,
            log_prob_floor=log_prob_floor,
        )
        results.append(result)
        print(f"[dataset-done] {dataset_name_from_path(path)} seconds={time.time() - started:.1f}", flush=True)
    aggregate_and_plot(results, args.output_dir)
    print(f"[done] outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
