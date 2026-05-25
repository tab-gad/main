from __future__ import annotations

import argparse
import gc
import importlib.util
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


REPO_ROOT = Path("/home/haeylee/spade")
REAL03_SCRIPT_PATH = REPO_ROOT / "experiments/dag/adbench_benchmark/run_real_03_density_style_eval.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/conditional_density/results/order_stability"


def load_real03_module():
    spec = importlib.util.spec_from_file_location("real03_module", REAL03_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {REAL03_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


real03 = load_real03_module()
DATASET_REGISTRY = real03.DATASET_REGISTRY


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_df(df: pd.DataFrame, max_rows: int | None, random_seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)


def build_full_chain_scheme_from_order(feature_order: list[str]) -> list[tuple[str, list[str]]]:
    return [(target, feature_order[:idx]) for idx, target in enumerate(feature_order)]


def build_order_bank(
    feature_names: list[str],
    *,
    n_random_orders: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    canonical = list(feature_names)
    reverse = list(reversed(feature_names))
    orders: list[dict[str, Any]] = [
        {
            "order_name": "canonical_order",
            "order_type": "canonical",
            "feature_order": canonical,
        },
        {
            "order_name": "reverse_order",
            "order_type": "reverse",
            "feature_order": reverse,
        },
    ]

    seen = {tuple(canonical), tuple(reverse)}
    rng = np.random.default_rng(random_seed)
    random_count = 0
    while random_count < n_random_orders:
        perm = tuple(rng.permutation(feature_names).tolist())
        if perm in seen:
            continue
        random_count += 1
        seen.add(perm)
        orders.append(
            {
                "order_name": f"random_order_{random_count:02d}",
                "order_type": "random",
                "feature_order": list(perm),
            }
        )
    return orders


def order_signature_from_steps(steps: list[tuple[str, list[str]]], limit: int = 8) -> str:
    parts = [f"{target}<-{len(parents)}" for target, parents in steps[:limit]]
    suffix = " ..." if len(steps) > limit else ""
    return " | ".join(parts) + suffix


def format_factor_terms(steps: list[tuple[str, list[str]]], limit: int = 5) -> list[str]:
    terms: list[str] = []
    for target, parents in steps[:limit]:
        if parents:
            terms.append(f"P({target} | {', '.join(parents)})")
        else:
            terms.append(f"P({target})")
    return terms


def make_unsupervised_wrapper(device: str, n_estimators: int, random_seed: int):
    return real03.make_unsupervised_wrapper(
        device=device,
        n_estimators=n_estimators,
        random_seed=random_seed,
    )


def build_step_matrix(df: pd.DataFrame, target: str, parents: list[str]) -> np.ndarray:
    columns = parents + [target]
    return df[columns].to_numpy(dtype=np.float32)


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"none", "null", "off"}:
        return None
    return float(value)


def fit_step_model_tabpfn(
    train_df: pd.DataFrame,
    target: str,
    parents: list[str],
    *,
    device: str,
    n_estimators: int,
    random_seed: int,
) -> dict[str, Any]:
    wrapper = make_unsupervised_wrapper(device=device, n_estimators=n_estimators, random_seed=random_seed)
    X_train_step = build_step_matrix(train_df, target, parents)
    started_at = time.time()
    wrapper.fit(X_train_step)
    wrapper_setup_seconds = time.time() - started_at
    return {
        "target": target,
        "parents": parents,
        "wrapper": wrapper,
        "wrapper_setup_seconds": wrapper_setup_seconds,
    }


def score_step_logspace(
    model_info: dict[str, Any],
    df: pd.DataFrame,
    *,
    log_prob_floor: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    target = model_info["target"]
    parents = model_info["parents"]
    wrapper = model_info["wrapper"]

    X_eval_step = build_step_matrix(df, target, parents)
    X_eval_tensor = torch.tensor(X_eval_step, dtype=torch.float32)
    conditional_idx = list(range(len(parents)))
    column_idx = len(parents)

    started_at = time.time()
    model, X_predict, y_predict = wrapper.density_(
        X_eval_tensor,
        wrapper.X_,
        conditional_idx,
        column_idx,
    )

    if wrapper.use_classifier_(column_idx, y_predict):
        pred_np = model.predict_proba(X_predict.numpy())
        y_indices = (
            y_predict.long()
            if torch.is_tensor(y_predict)
            else torch.tensor(y_predict, dtype=torch.long)
        )
        pred_tensor = torch.ones(len(y_indices), dtype=torch.float32) * 0.1
        valid_indices = (y_indices >= 0) & (y_indices < pred_np.shape[1])
        if valid_indices.any():
            for idx in torch.where(valid_indices)[0].tolist():
                y_idx = int(y_indices[idx])
                pred_tensor[idx] = float(pred_np[idx, y_idx])
        if log_prob_floor is not None:
            pred_tensor = torch.clamp(pred_tensor, min=log_prob_floor)
        log_score_tensor = torch.log(pred_tensor)
        branch = np.array(["classifier"] * len(pred_tensor), dtype=object)
        raw_value = pred_tensor.detach().cpu().numpy()
        out_of_support = np.full(len(pred_tensor), np.nan, dtype=np.float32)
    else:
        pred = model.predict(X_predict, output_type="full")
        logits = pred["logits"]
        logits_tensor = logits.clone().detach() if torch.is_tensor(logits) else torch.as_tensor(logits)
        y_tensor = y_predict.clone().detach().to(logits_tensor.device)
        criterion = pred["criterion"]
        nll_tensor = criterion.forward(logits_tensor, y_tensor)
        log_score_tensor = -nll_tensor.to(torch.float32)
        if log_prob_floor is not None:
            log_score_tensor = torch.clamp(log_score_tensor, min=float(np.log(log_prob_floor)))
        branch = np.array(["regressor"] * len(log_score_tensor), dtype=object)
        raw_value = log_score_tensor.detach().cpu().numpy()
        borders = criterion.borders.detach().cpu().numpy()
        y_np = y_tensor.detach().cpu().numpy()
        out_of_support = ((y_np < borders[0]) | (y_np > borders[-1])).astype(np.float32)

    conditional_fit_and_score_seconds = time.time() - started_at
    return (
        log_score_tensor.detach().cpu().numpy(),
        raw_value,
        out_of_support,
        conditional_fit_and_score_seconds,
    )


def fit_and_score_order(
    order_name: str,
    order_type: str,
    steps: list[tuple[str, list[str]]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    device: str,
    n_estimators: int,
    random_seed: int,
    log_prob_floor: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_df = test_df.copy()
    timing_rows: list[dict[str, Any]] = []
    total_wrapper_setup_seconds = 0.0
    total_conditional_fit_and_score_seconds = 0.0
    total_out_of_support_hits = 0.0

    for step_idx, (target, parents) in enumerate(steps, start=1):
        model_info = fit_step_model_tabpfn(
            train_df=train_df,
            target=target,
            parents=parents,
            device=device,
            n_estimators=n_estimators,
            random_seed=random_seed,
        )
        log_score, raw_value, out_of_support, conditional_fit_and_score_seconds = score_step_logspace(
            model_info,
            eval_df,
            log_prob_floor=log_prob_floor,
        )
        wrapper_setup_seconds = model_info["wrapper_setup_seconds"]
        total_wrapper_setup_seconds += wrapper_setup_seconds
        total_conditional_fit_and_score_seconds += conditional_fit_and_score_seconds
        valid_oos = out_of_support[~np.isnan(out_of_support)]
        total_out_of_support_hits += float(valid_oos.sum()) if len(valid_oos) else 0.0

        eval_df[f"{order_name}__{target}__log_score"] = log_score
        eval_df[f"{order_name}__{target}__raw_value"] = raw_value
        eval_df[f"{order_name}__{target}__out_of_support"] = out_of_support

        timing_rows.append(
            {
                "order_name": order_name,
                "order_type": order_type,
                "step_idx": step_idx,
                "target": target,
                "n_parents": len(parents),
                "wrapper_setup_seconds": wrapper_setup_seconds,
                "conditional_fit_and_score_seconds": conditional_fit_and_score_seconds,
                "total_runtime_seconds": wrapper_setup_seconds + conditional_fit_and_score_seconds,
                "n_out_of_support_hits": float(valid_oos.sum()) if len(valid_oos) else np.nan,
                "out_of_support_rate": float(valid_oos.mean()) if len(valid_oos) else np.nan,
            }
        )

    log_score_cols = [f"{order_name}__{target}__log_score" for target, _ in steps]
    eval_df[f"{order_name}__total_log_score"] = eval_df[log_score_cols].sum(axis=1)

    metrics = {
        "order_name": order_name,
        "order_type": order_type,
        "n_steps": len(steps),
        "avg_parents": float(np.mean([len(parents) for _, parents in steps])),
        "max_parents": int(max([len(parents) for _, parents in steps], default=0)),
        "total_wrapper_setup_seconds": total_wrapper_setup_seconds,
        "total_conditional_fit_and_score_seconds": total_conditional_fit_and_score_seconds,
        "total_runtime_seconds": total_wrapper_setup_seconds + total_conditional_fit_and_score_seconds,
        "mean_total_log_score": float(eval_df[f"{order_name}__total_log_score"].mean()),
        "std_total_log_score": float(eval_df[f"{order_name}__total_log_score"].std(ddof=0)),
        "min_total_log_score": float(eval_df[f"{order_name}__total_log_score"].min()),
        "max_total_log_score": float(eval_df[f"{order_name}__total_log_score"].max()),
        "total_out_of_support_hits": total_out_of_support_hits,
        "order_signature": order_signature_from_steps(steps),
        "log_prob_floor": log_prob_floor if log_prob_floor is not None else np.nan,
    }
    release_runtime_memory()
    return eval_df, pd.DataFrame(timing_rows), metrics


def build_sample_score_frame(
    eval_df: pd.DataFrame,
    *,
    dataset_name: str,
    order_name: str,
    order_type: str,
    score_col: str,
) -> pd.DataFrame:
    out_df = eval_df[["test_index", "source_type", "y_test_binary"]].copy()
    out_df["dataset_name"] = dataset_name
    out_df["order_name"] = order_name
    out_df["order_type"] = order_type
    out_df["total_log_score"] = eval_df[score_col].to_numpy(dtype=np.float32)
    return out_df[
        [
            "dataset_name",
            "order_name",
            "order_type",
            "test_index",
            "source_type",
            "y_test_binary",
            "total_log_score",
        ]
    ]


def summarize_sample_stability(sample_scores_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sample_scores_df.groupby(["dataset_name", "test_index", "source_type", "y_test_binary"])["total_log_score"]
        .agg(
            mean_log_score="mean",
            std_log_score=lambda x: float(np.std(x.to_numpy(dtype=np.float64), ddof=0)),
            min_log_score="min",
            max_log_score="max",
        )
        .reset_index()
    )
    grouped["range_log_score"] = grouped["max_log_score"] - grouped["min_log_score"]
    return grouped


def summarize_dataset_stability(
    sample_stability_df: pd.DataFrame,
    *,
    dataset_name: str,
    n_orders_tested: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_type, sub_df in [
        ("all", sample_stability_df),
        ("normal", sample_stability_df[sample_stability_df["source_type"] == "normal"]),
        ("anomaly", sample_stability_df[sample_stability_df["source_type"] == "anomaly"]),
    ]:
        if sub_df.empty:
            continue
        rows.append(
            {
                "dataset_name": dataset_name,
                "source_group": source_type,
                "n_samples": int(len(sub_df)),
                "n_orders_tested": int(n_orders_tested),
                "mean_std_log_score": float(sub_df["std_log_score"].mean()),
                "median_std_log_score": float(sub_df["std_log_score"].median()),
                "p90_std_log_score": float(sub_df["std_log_score"].quantile(0.9)),
                "max_std_log_score": float(sub_df["std_log_score"].max()),
                "mean_range_log_score": float(sub_df["range_log_score"].mean()),
                "median_range_log_score": float(sub_df["range_log_score"].median()),
                "p90_range_log_score": float(sub_df["range_log_score"].quantile(0.9)),
                "max_range_log_score": float(sub_df["range_log_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_order_rank_correlation(sample_scores_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    pivot = sample_scores_df.pivot(index="test_index", columns="order_name", values="total_log_score").sort_index()
    order_names = pivot.columns.tolist()
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(order_names):
        for right in order_names[i + 1 :]:
            rho, _ = spearmanr(pivot[left], pivot[right])
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "left_order": left,
                    "right_order": right,
                    "spearman_rho": float(rho),
                }
            )
    return pd.DataFrame(rows)


def summarize_rank_correlation(pairwise_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame(
            [
                {
                    "dataset_name": dataset_name,
                    "mean_spearman_rho": np.nan,
                    "median_spearman_rho": np.nan,
                    "min_spearman_rho": np.nan,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "mean_spearman_rho": float(pairwise_df["spearman_rho"].mean()),
                "median_spearman_rho": float(pairwise_df["spearman_rho"].median()),
                "min_spearman_rho": float(pairwise_df["spearman_rho"].min()),
            }
        ]
    )


def trace_sample_through_order(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_order: list[str],
    *,
    sample_row_idx: int,
    device: str,
    n_estimators: int,
    random_seed: int,
    max_steps: int = 6,
    log_prob_floor: float | None = 1e-10,
) -> pd.DataFrame:
    steps = build_full_chain_scheme_from_order(feature_order)
    rows: list[dict[str, Any]] = []
    sample_df = test_df.iloc[[sample_row_idx]].copy()

    for step_idx, (target, parents) in enumerate(steps[:max_steps], start=1):
        model_info = fit_step_model_tabpfn(
            train_df=train_df,
            target=target,
            parents=parents,
            device=device,
            n_estimators=n_estimators,
            random_seed=random_seed,
        )
        wrapper = model_info["wrapper"]
        X_eval_step = build_step_matrix(sample_df, target, parents)
        X_eval_tensor = torch.tensor(X_eval_step, dtype=torch.float32)
        conditional_idx = list(range(len(parents)))
        column_idx = len(parents)
        model, X_predict, y_predict = wrapper.density_(
            X_eval_tensor,
            wrapper.X_,
            conditional_idx,
            column_idx,
        )

        if wrapper.use_classifier_(column_idx, y_predict):
            pred_np = model.predict_proba(X_predict.numpy())
            y_idx = int(y_predict.long()[0])
            used_pred = float(pred_np[0, y_idx])
            if log_prob_floor is not None:
                used_pred = max(used_pred, log_prob_floor)
            local_log_score = float(np.log(used_pred))
            rows.append(
                {
                    "step_idx": step_idx,
                    "target": target,
                    "parents": ", ".join(parents),
                    "factor_term": f"P({target})" if not parents else f"P({target} | {', '.join(parents)})",
                    "branch": "classifier",
                    "y_value": float(y_predict[0].detach().cpu()),
                    "support_left": np.nan,
                    "support_right": np.nan,
                    "out_of_support": np.nan,
                    "local_log_score": local_log_score,
                }
            )
        else:
            pred = model.predict(X_predict, output_type="full")
            logits_tensor = pred["logits"].clone().detach()
            y_tensor = y_predict.clone().detach().to(logits_tensor.device)
            criterion = pred["criterion"]
            nll_tensor = criterion.forward(logits_tensor, y_tensor)
            local_log_score = float((-nll_tensor)[0].detach().cpu())
            if log_prob_floor is not None:
                local_log_score = max(local_log_score, float(np.log(log_prob_floor)))
            borders = criterion.borders.detach().cpu().numpy()
            y_value = float(y_tensor[0].detach().cpu())
            rows.append(
                {
                    "step_idx": step_idx,
                    "target": target,
                    "parents": ", ".join(parents),
                    "factor_term": f"P({target})" if not parents else f"P({target} | {', '.join(parents)})",
                    "branch": "regressor",
                    "y_value": y_value,
                    "support_left": float(borders[0]),
                    "support_right": float(borders[-1]),
                    "out_of_support": float(y_value < borders[0] or y_value > borders[-1]),
                    "local_log_score": local_log_score,
                }
            )
        release_runtime_memory()

    trace_df = pd.DataFrame(rows)
    if not trace_df.empty:
        trace_df["cumulative_log_score"] = trace_df["local_log_score"].cumsum()
    return trace_df


def run_dataset(
    dataset_name: str,
    *,
    output_dir: Path,
    device: str,
    n_estimators: int,
    random_seed: int,
    test_size: float,
    n_random_orders: int,
    max_tabpfn_train_samples: int | None,
    selected_order_names: list[str] | None = None,
    output_prefix: str | None = None,
    log_prob_floor: float | None = 1e-10,
) -> dict[str, pd.DataFrame]:
    set_global_seed(random_seed)
    bundle = real03.load_dataset_bundle(dataset_name)
    split_bundle = real03.build_anomaly_split(bundle, test_size=test_size, random_seed=random_seed)
    train_df = sample_df(split_bundle["train_df"], max_tabpfn_train_samples, random_seed)
    test_df = split_bundle["test_df"].copy()
    orders = build_order_bank(bundle["feature_names"], n_random_orders=n_random_orders, random_seed=random_seed)
    if selected_order_names is not None:
        selected_set = set(selected_order_names)
        orders = [order for order in orders if order["order_name"] in selected_set]
        missing = sorted(selected_set - {order["order_name"] for order in orders})
        if missing:
            raise ValueError(f"Unknown order names requested: {missing}")
        if not orders:
            raise ValueError("No orders left after applying selected_order_names filter.")

    order_metric_rows: list[dict[str, Any]] = []
    order_manifest_rows: list[dict[str, Any]] = []
    timing_frames: list[pd.DataFrame] = []
    sample_score_frames: list[pd.DataFrame] = []

    for order_meta in orders:
        order_name = order_meta["order_name"]
        order_type = order_meta["order_type"]
        feature_order = order_meta["feature_order"]
        steps = build_full_chain_scheme_from_order(feature_order)
        print(f"[run] dataset={dataset_name} order={order_name}")

        eval_df, timing_df, metrics = fit_and_score_order(
            order_name=order_name,
            order_type=order_type,
            steps=steps,
            train_df=train_df,
            test_df=test_df,
            device=device,
            n_estimators=n_estimators,
            random_seed=random_seed,
            log_prob_floor=log_prob_floor,
        )
        order_metric_rows.append({"dataset_name": dataset_name, **metrics})
        order_manifest_rows.append(
            {
                "dataset_name": dataset_name,
                "order_name": order_name,
                "order_type": order_type,
                "n_steps": len(steps),
                "avg_parents": float(np.mean([len(parents) for _, parents in steps])),
                "max_parents": int(max([len(parents) for _, parents in steps], default=0)),
                "order_signature": order_signature_from_steps(steps),
                "first_factor_terms": " | ".join(format_factor_terms(steps)),
            }
        )
        timing_frames.append(timing_df.assign(dataset_name=dataset_name))
        sample_score_frames.append(
            build_sample_score_frame(
                eval_df,
                dataset_name=dataset_name,
                order_name=order_name,
                order_type=order_type,
                score_col=f"{order_name}__total_log_score",
            )
        )

    order_metrics_df = pd.DataFrame(order_metric_rows)
    order_manifest_df = pd.DataFrame(order_manifest_rows)
    timing_df = pd.concat(timing_frames, axis=0).reset_index(drop=True) if timing_frames else pd.DataFrame()
    sample_scores_df = pd.concat(sample_score_frames, axis=0).reset_index(drop=True) if sample_score_frames else pd.DataFrame()
    sample_stability_df = summarize_sample_stability(sample_scores_df)
    stability_summary_df = summarize_dataset_stability(
        sample_stability_df,
        dataset_name=dataset_name,
        n_orders_tested=len(orders),
    )
    rank_corr_pairwise_df = build_order_rank_correlation(sample_scores_df, dataset_name=dataset_name)
    rank_corr_summary_df = summarize_rank_correlation(rank_corr_pairwise_df, dataset_name=dataset_name)
    top_unstable_df = sample_stability_df.sort_values("std_log_score", ascending=False).head(25).reset_index(drop=True)

    manifest_df = split_bundle["manifest_df"].copy()
    manifest_df["n_orders_tested"] = len(orders)
    manifest_df["train_normal_count_after_cap"] = len(train_df)
    manifest_df["max_tabpfn_train_samples"] = max_tabpfn_train_samples if max_tabpfn_train_samples is not None else np.nan
    manifest_df["n_random_orders"] = n_random_orders
    manifest_df["device"] = device
    manifest_df["n_estimators"] = n_estimators
    manifest_df["log_prob_floor"] = log_prob_floor if log_prob_floor is not None else np.nan

    prefix = output_prefix if output_prefix is not None else dataset_name

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_dir / f"{prefix}_dataset_manifest.csv", index=False)
    order_manifest_df.to_csv(output_dir / f"{prefix}_order_manifest.csv", index=False)
    order_metrics_df.to_csv(output_dir / f"{prefix}_order_metrics.csv", index=False)
    timing_df.to_csv(output_dir / f"{prefix}_timing_detail.csv", index=False)
    sample_scores_df.to_csv(output_dir / f"{prefix}_sample_scores.csv", index=False)
    sample_stability_df.to_csv(output_dir / f"{prefix}_sample_stability.csv", index=False)
    stability_summary_df.to_csv(output_dir / f"{prefix}_stability_summary.csv", index=False)
    rank_corr_pairwise_df.to_csv(output_dir / f"{prefix}_rank_corr_pairwise.csv", index=False)
    rank_corr_summary_df.to_csv(output_dir / f"{prefix}_rank_corr_summary.csv", index=False)
    top_unstable_df.to_csv(output_dir / f"{prefix}_top_unstable_samples.csv", index=False)

    return {
        "dataset_manifest_df": manifest_df,
        "order_manifest_df": order_manifest_df,
        "order_metrics_df": order_metrics_df,
        "timing_df": timing_df,
        "sample_scores_df": sample_scores_df,
        "sample_stability_df": sample_stability_df,
        "stability_summary_df": stability_summary_df,
        "rank_corr_pairwise_df": rank_corr_pairwise_df,
        "rank_corr_summary_df": rank_corr_summary_df,
        "top_unstable_df": top_unstable_df,
    }


def aggregate_batch_results(batch_results: list[dict[str, pd.DataFrame]], output_dir: Path) -> None:
    key_to_filename = {
        "dataset_manifest_df": "batch_dataset_manifest.csv",
        "order_manifest_df": "batch_order_manifest.csv",
        "order_metrics_df": "batch_order_metrics.csv",
        "timing_df": "batch_timing_detail.csv",
        "sample_scores_df": "batch_sample_scores.csv",
        "sample_stability_df": "batch_sample_stability.csv",
        "stability_summary_df": "batch_stability_summary.csv",
        "rank_corr_pairwise_df": "batch_rank_corr_pairwise.csv",
        "rank_corr_summary_df": "batch_rank_corr_summary.csv",
        "top_unstable_df": "batch_top_unstable_samples.csv",
    }
    for key, filename in key_to_filename.items():
        frames = [result[key] for result in batch_results if not result[key].empty]
        if not frames:
            continue
        pd.concat(frames, axis=0).reset_index(drop=True).to_csv(output_dir / filename, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Order stability experiment for conditional density TabPFN full-chain scoring.")
    parser.add_argument("--dataset", type=str, default="WDBC", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--run-batch", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--n-random-orders", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--max-tabpfn-train-samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--order-name", type=str, action="append", default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--log-prob-floor", type=str, default="1e-10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_prob_floor = parse_optional_float(args.log_prob_floor)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Selected device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_batch:
        batch_results = []
        for dataset_name in DATASET_REGISTRY:
            batch_results.append(
                run_dataset(
                    dataset_name,
                    output_dir=args.output_dir,
                    device=device,
                    n_estimators=args.n_estimators,
                    random_seed=args.random_seed,
                    test_size=args.test_size,
                    n_random_orders=args.n_random_orders,
                    max_tabpfn_train_samples=args.max_tabpfn_train_samples,
                    selected_order_names=args.order_name,
                    output_prefix=args.output_prefix,
                    log_prob_floor=log_prob_floor,
                )
            )
        aggregate_batch_results(batch_results, args.output_dir)
        print(f"Saved batch outputs to {args.output_dir}")
    else:
        run_dataset(
            args.dataset,
            output_dir=args.output_dir,
            device=device,
            n_estimators=args.n_estimators,
            random_seed=args.random_seed,
            test_size=args.test_size,
            n_random_orders=args.n_random_orders,
            max_tabpfn_train_samples=args.max_tabpfn_train_samples,
            selected_order_names=args.order_name,
            output_prefix=args.output_prefix,
            log_prob_floor=log_prob_floor,
        )
        print(f"Saved dataset outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
