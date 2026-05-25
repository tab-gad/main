"""check_seed_sensitivity.py — seed 만 바꿀 때 mean_std_log_score 가 얼마나 흔들리나.

==============================================================================
목적
==============================================================================
HY 와 KS 결과가 cardio, fault 등에서 ~1-2 정도 차이가 났음.
같은 setting (22 ordering, train cap 1000, BarDistribution) 인데 seed 만
다르게 하면 mean_std_log_score 가 얼마나 흔들리는지 직접 측정.

차이가 작으면 (예: ~0.1) → HY vs KS 차이는 seed 외 다른 원인.
차이가 크면 (~1-2) → seed 변동만으로 그 차이 설명 가능.

==============================================================================
방법
==============================================================================
1. 검증할 dataset: cardio, fault (HY 와 차이 큰 것).
2. 다른 seed 5개 (42, 0, 1, 2, 3) 로 각각 run_one 실행.
3. mean_std_log_score / mean_spearman_rho 변동 출력.

==============================================================================
실행
==============================================================================
python check_seed_sensitivity.py
==============================================================================
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import ks_helpers as kh
from ks_helpers import (DATA_DIR, load_npz, build_split, cap_train,
                          build_orders, total_log_score, reset_cache,
                          set_global_seed)

# ==== 검증 ====
TARGET_DATASETS = ['6_cardio.npz', '12_fault.npz']
SEEDS = [42, 0, 1, 2, 3]


def measure(name, X, y, seed):
    """한 dataset, 한 seed 로 mean_std_log_score 등 측정."""
    # ks_helpers 모듈 안의 build_orders / build_split / cap_train 이
    # 모두 RANDOM_SEED 를 default 로 받음 → 시드만 인자로 전달.
    set_global_seed(seed)
    reset_cache()
    X_tr, X_te, y_te = build_split(X, y, seed=seed)
    X_tr = cap_train(X_tr, seed=seed)
    p = X.shape[1]
    orders = build_orders(p, seed=seed)
    print(f'  [{name} seed={seed}] p={p} n_tr={len(X_tr)} n_te={len(X_te)} M={len(orders)}')

    scores = []
    for _, ordering in orders:
        s = total_log_score(X_tr, X_te, ordering, seed=seed)
        scores.append(s)
    S = np.stack(scores, axis=0)                # (M, n_te)
    per_sample_std = S.std(axis=0)               # (n_te,)

    # rank correlation
    M = len(orders); rhos = []
    for i in range(M):
        for j in range(i + 1, M):
            rho, _ = spearmanr(S[i], S[j])
            rhos.append(rho)
    return {
        'mean_std_log_score': float(per_sample_std.mean()),
        'p90_std_log_score':  float(np.quantile(per_sample_std, 0.90)),
        'mean_spearman_rho':  float(np.mean(rhos)),
        'min_spearman_rho':   float(np.min(rhos)),
    }


def main():
    rows = []
    for fname in TARGET_DATASETS:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f'{fname}: missing'); continue
        name = fname.replace('.npz', '')
        X, y = load_npz(path)
        print(f'\n=== {name} (p={X.shape[1]}) ===')
        for seed in SEEDS:
            t0 = time.time()
            try:
                m = measure(name, X, y, seed)
                m.update({'dataset': name, 'seed': seed,
                          'sec': round(time.time() - t0, 1)})
                rows.append(m)
                print(f'    → mean_std={m["mean_std_log_score"]:.3f}  '
                      f'p90={m["p90_std_log_score"]:.3f}  '
                      f'rho_mean={m["mean_spearman_rho"]:.3f}  '
                      f'({m["sec"]}s)', flush=True)
            except Exception as e:
                print(f'    FAIL seed={seed}: {type(e).__name__}: {e}')

    df = pd.DataFrame(rows)
    out = './KS_results/seed_sensitivity.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\nsaved {out}')

    # 요약: dataset 별 seed-간 std
    if not df.empty:
        print('\n=== summary (seed 간 변동) ===')
        summary = df.groupby('dataset').agg(
            mean=('mean_std_log_score', 'mean'),
            std=('mean_std_log_score', 'std'),
            min=('mean_std_log_score', 'min'),
            max=('mean_std_log_score', 'max'),
            rho_mean=('mean_spearman_rho', 'mean'),
            rho_std=('mean_spearman_rho', 'std'),
        )
        print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
