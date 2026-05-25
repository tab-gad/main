from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("POSTHOG_DISABLED", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import adbench_order_sensitivity_screening as screening
import local_order_decomposition as lod
import order_stability_batch as osb


DEFAULT_OUTPUT_DIR = (
    osb.REPO_ROOT
    / "experiments/conditional_density/results/adbench_local_order_decomposition_2026_05_18"
)


DEFAULT_DATASETS = [
    # High sensitivity in the lightweight ADBench screening.
    "fault",
    "letter",
    "cardio",
    "Cardiotocography",
    "Ionosphere",
    # Medium / mixed sensitivity.
    "pendigits",
    "PageBlocks",
    "WPBC",
    "WDBC",
    # Low sensitivity controls.
    "Waveform",
    "yeast",
    "annthyroid",
]


def score_dataset(
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
    bundle = screening.load_npz_bundle(path)
    split = screening.build_split(bundle, test_size=test_size, random_seed=random_seed)
    train_df_full = split["train_df"]
    train_df = osb.sample_df(train_df_full, max_tabpfn_train_samples, random_seed)
    test_df = split["test_df"].copy()
    orders = screening.select_orders(
        bundle["feature_names"],
        n_random_orders=n_random_orders,
        random_seed=random_seed,
    )

    local_frames: list[pd.DataFrame] = []
    timing_frames: list[pd.DataFrame] = []
    order_rows: list[dict[str, Any]] = []
    dataset_name = bundle["dataset_name"]

    for order_meta in orders:
        print(f"[run] dataset={dataset_name} order={order_meta['order_name']}", flush=True)
        local_df, timing_df = lod.score_order_local_terms(
            dataset_name=dataset_name,
            order_meta=order_meta,
            train_df=train_df,
            test_df=test_df,
            device=device,
            n_estimators=n_estimators,
            random_seed=random_seed,
            log_prob_floor=log_prob_floor,
        )
        local_frames.append(local_df)
        timing_frames.append(timing_df)
        order_rows.append(
            {
                "dataset_name": dataset_name,
                "order_name": order_meta["order_name"],
                "order_type": order_meta["order_type"],
                "n_features": len(bundle["feature_names"]),
                "feature_order": " | ".join(order_meta["feature_order"]),
            }
        )
        osb.release_runtime_memory()

    local_all = pd.concat(local_frames, ignore_index=True)
    timing_all = pd.concat(timing_frames, ignore_index=True)
    order_manifest = pd.DataFrame(order_rows)
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
    for key, value in screening.safe_corr_stats(train_df_full).items():
        manifest[key] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    local_all.to_csv(output_dir / f"{dataset_name}_local_terms.csv", index=False)
    timing_all.to_csv(output_dir / f"{dataset_name}_timing.csv", index=False)
    order_manifest.to_csv(output_dir / f"{dataset_name}_order_manifest.csv", index=False)
    manifest.to_csv(output_dir / f"{dataset_name}_dataset_manifest.csv", index=False)

    return {
        "local_terms": local_all,
        "timing": timing_all,
        "order_manifest": order_manifest,
        "manifest": manifest,
    }


def build_total_score_outputs(local_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    finite = local_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["local_log_score"]).copy()
    total_scores = (
        finite.groupby(
            [
                "dataset_name",
                "order_name",
                "order_type",
                "test_index",
                "source_type",
                "y_test_binary",
            ],
            as_index=False,
        )["local_log_score"]
        .sum()
        .rename(columns={"local_log_score": "total_log_score"})
    )

    sample_stability = osb.summarize_sample_stability(total_scores)
    stability_frames: list[pd.DataFrame] = []
    rank_pairwise_frames: list[pd.DataFrame] = []
    rank_summary_frames: list[pd.DataFrame] = []
    for dataset_name, sub_scores in total_scores.groupby("dataset_name"):
        n_orders = sub_scores["order_name"].nunique()
        sub_stability = sample_stability[sample_stability["dataset_name"] == dataset_name]
        stability_frames.append(
            osb.summarize_dataset_stability(
                sub_stability,
                dataset_name=dataset_name,
                n_orders_tested=n_orders,
            )
        )
        rank_pairwise = osb.build_order_rank_correlation(sub_scores, dataset_name=dataset_name)
        rank_pairwise_frames.append(rank_pairwise)
        rank_summary_frames.append(osb.summarize_rank_correlation(rank_pairwise, dataset_name=dataset_name))

    return {
        "sample_scores": total_scores,
        "sample_stability": sample_stability,
        "stability_summary": pd.concat(stability_frames, ignore_index=True),
        "rank_corr_pairwise": pd.concat(rank_pairwise_frames, ignore_index=True),
        "rank_corr_summary": pd.concat(rank_summary_frames, ignore_index=True),
    }


def summarize_feature_concentration(feature_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, sub in feature_summary.groupby("dataset_name"):
        vals = sub["mean_local_std"].dropna().sort_values(ascending=False).to_numpy(dtype=float)
        total = float(vals.sum())
        rows.append(
            {
                "dataset_name": dataset_name,
                "n_features": int(len(vals)),
                "mean_feature_local_std": float(vals.mean()) if len(vals) else np.nan,
                "median_feature_local_std": float(np.median(vals)) if len(vals) else np.nan,
                "max_feature_local_std": float(vals[0]) if len(vals) else np.nan,
                "top1_share_of_sum_local_std": float(vals[:1].sum() / total) if total > 0 else np.nan,
                "top3_share_of_sum_local_std": float(vals[:3].sum() / total) if total > 0 else np.nan,
                "top5_share_of_sum_local_std": float(vals[:5].sum() / total) if total > 0 else np.nan,
                "n_features_mean_local_std_gt_1": int((vals > 1.0).sum()),
                "n_features_mean_local_std_gt_1_5": int((vals > 1.5).sum()),
                "n_features_mean_local_std_gt_2": int((vals > 2.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_dataset_metric_summary(
    *,
    manifest: pd.DataFrame,
    stability_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    feature_concentration: pd.DataFrame,
    position_summary: pd.DataFrame,
) -> pd.DataFrame:
    all_stab = stability_summary[stability_summary["source_group"] == "all"].copy()
    normal = stability_summary[stability_summary["source_group"] == "normal"][
        ["dataset_name", "mean_std_log_score", "p90_std_log_score"]
    ].rename(
        columns={
            "mean_std_log_score": "normal_mean_std_log_score",
            "p90_std_log_score": "normal_p90_std_log_score",
        }
    )
    anomaly = stability_summary[stability_summary["source_group"] == "anomaly"][
        ["dataset_name", "mean_std_log_score", "p90_std_log_score"]
    ].rename(
        columns={
            "mean_std_log_score": "anomaly_mean_std_log_score",
            "p90_std_log_score": "anomaly_p90_std_log_score",
        }
    )
    pos = (
        position_summary.groupby("dataset_name")
        .agg(
            median_abs_position_spearman=("abs_position_spearman", "median"),
            p90_abs_position_spearman=("abs_position_spearman", lambda x: float(np.nanquantile(x, 0.9))),
            n_features_abs_position_spearman_gt_0_5=(
                "abs_position_spearman",
                lambda x: int((x.dropna() >= 0.5).sum()),
            ),
        )
        .reset_index()
    )
    summary = (
        manifest.merge(all_stab, on="dataset_name", how="left")
        .merge(normal, on="dataset_name", how="left")
        .merge(anomaly, on="dataset_name", how="left")
        .merge(rank_summary, on="dataset_name", how="left")
        .merge(feature_concentration, on="dataset_name", how="left", suffixes=("", "_feature_conc"))
        .merge(pos, on="dataset_name", how="left")
    )
    summary["ranking_instability"] = 1.0 - summary["mean_spearman_rho"]
    summary["anomaly_minus_normal_mean_std"] = (
        summary["anomaly_mean_std_log_score"] - summary["normal_mean_std_log_score"]
    )
    return summary.sort_values("mean_std_log_score", ascending=False).reset_index(drop=True)


def plot_summary_outputs(
    *,
    dataset_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = dataset_summary.sort_values("mean_std_log_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.38 * len(ranked))))
    ax.barh(ranked["dataset_name"], ranked["mean_std_log_score"], color="#64748B")
    ax.set_xlabel("total mean std_log_score across 22 orders")
    ax.set_title("Dataset-level order sensitivity")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_total_mean_std_ranked.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.scatter(
        dataset_summary["mean_std_log_score"],
        1.0 - dataset_summary["mean_spearman_rho"],
        s=70,
        alpha=0.82,
        color="#2563EB",
    )
    for _, row in dataset_summary.iterrows():
        ax.annotate(
            row["dataset_name"],
            (row["mean_std_log_score"], 1.0 - row["mean_spearman_rho"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("total mean std_log_score")
    ax.set_ylabel("1 - mean Spearman rho")
    ax.set_title("Score instability vs ranking instability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "score_vs_ranking_instability.png", dpi=180)
    plt.close(fig)

    top = feature_summary[feature_summary["instability_rank"] <= 5].copy()
    datasets = dataset_summary["dataset_name"].tolist()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10.5, 2.1 * len(datasets)), squeeze=False)
    for ax, dataset_name in zip(axes.ravel(), datasets):
        sub = top[top["dataset_name"] == dataset_name].sort_values("mean_local_std")
        ax.barh(sub["target"], sub["mean_local_std"], color="#F97316", alpha=0.82)
        ax.set_title(f"{dataset_name}: top local order-sensitive features")
        ax.set_xlabel("mean local std")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "top5_local_order_sensitive_features.png", dpi=180)
    plt.close(fig)


def aggregate_and_write(results: list[dict[str, pd.DataFrame]], output_dir: Path) -> None:
    local_all = pd.concat([r["local_terms"] for r in results], ignore_index=True)
    timing_all = pd.concat([r["timing"] for r in results], ignore_index=True)
    orders_all = pd.concat([r["order_manifest"] for r in results], ignore_index=True)
    manifest_all = pd.concat([r["manifest"] for r in results], ignore_index=True)

    feature_summary, sample_feature_summary = lod.summarize_feature_instability(local_all)
    position_summary = lod.summarize_position_effect(local_all)
    total_outputs = build_total_score_outputs(local_all)
    feature_concentration = summarize_feature_concentration(feature_summary)
    dataset_summary = build_dataset_metric_summary(
        manifest=manifest_all,
        stability_summary=total_outputs["stability_summary"],
        rank_summary=total_outputs["rank_corr_summary"],
        feature_concentration=feature_concentration,
        position_summary=position_summary,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    local_all.to_csv(output_dir / "batch_local_terms.csv", index=False)
    timing_all.to_csv(output_dir / "batch_timing.csv", index=False)
    orders_all.to_csv(output_dir / "batch_order_manifest.csv", index=False)
    manifest_all.to_csv(output_dir / "batch_dataset_manifest.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_instability_summary.csv", index=False)
    sample_feature_summary.to_csv(output_dir / "sample_feature_instability.csv", index=False)
    position_summary.to_csv(output_dir / "position_effect_summary.csv", index=False)
    feature_concentration.to_csv(output_dir / "feature_concentration_summary.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_metric_summary.csv", index=False)
    for name, df in total_outputs.items():
        df.to_csv(output_dir / f"batch_{name}.csv", index=False)

    plot_summary_outputs(
        dataset_summary=dataset_summary,
        feature_summary=feature_summary,
        output_dir=output_dir,
    )


def load_existing_results(dataset_names: list[str], output_dir: Path) -> list[dict[str, pd.DataFrame]]:
    results: list[dict[str, pd.DataFrame]] = []
    missing: list[str] = []
    for dataset_name in dataset_names:
        paths = {
            "local_terms": output_dir / f"{dataset_name}_local_terms.csv",
            "timing": output_dir / f"{dataset_name}_timing.csv",
            "order_manifest": output_dir / f"{dataset_name}_order_manifest.csv",
            "manifest": output_dir / f"{dataset_name}_dataset_manifest.csv",
        }
        if not all(path.exists() for path in paths.values()):
            missing.append(dataset_name)
            continue
        results.append({key: pd.read_csv(path) for key, path in paths.items()})
    if missing:
        raise FileNotFoundError(f"Missing per-dataset outputs for: {', '.join(missing)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ADBench feature-level local order decomposition.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda-device-index", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--n-random-orders", type=int, default=20)
    parser.add_argument("--max-tabpfn-train-samples", type=int, default=1000)
    parser.add_argument("--log-prob-floor", type=str, default="1e-10")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()

    log_prob_floor = osb.parse_optional_float(args.log_prob_floor)
    device = "cuda" if args.device == "auto" and osb.torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and args.cuda_device_index is not None:
        osb.torch.cuda.set_device(args.cuda_device_index)
        device = f"cuda:{args.cuda_device_index}"

    paths = screening.resolve_dataset_paths(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device}", flush=True)
    if device.startswith("cuda"):
        print(f"[info] cuda_current_device={osb.torch.cuda.current_device()}", flush=True)
    print(f"[info] n_datasets={len(paths)} n_orders={args.n_random_orders + 2}", flush=True)

    if args.aggregate_existing:
        dataset_names = [screening.dataset_name_from_path(path) for path in paths]
        existing_results = load_existing_results(dataset_names, args.output_dir)
        aggregate_and_write(existing_results, args.output_dir)
        print(f"[done] aggregated existing outputs in {args.output_dir}", flush=True)
        return

    results: list[dict[str, pd.DataFrame]] = []
    for path in paths:
        started = time.time()
        dataset_name = screening.dataset_name_from_path(path)
        print(f"[dataset] {dataset_name} path={path.name}", flush=True)
        result = score_dataset(
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
        print(f"[dataset-done] {dataset_name} seconds={time.time() - started:.1f}", flush=True)

    if not args.no_aggregate:
        aggregate_and_write(results, args.output_dir)
        print(f"[done] outputs saved to {args.output_dir}", flush=True)
    else:
        print(f"[done] per-dataset outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
