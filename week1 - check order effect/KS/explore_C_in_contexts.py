"""explore_C_in_contexts.py — 한 변수 C 가 prefix 바뀔 때 얼마나 흔들리는지.

==============================================================================
목적
==============================================================================
같은 변수 C 라도 ordering 마다 다른 prefix 에서 평가됨.
    ord_01: log p(C | A, B)
    ord_02: log p(C | D, A, B)
    ord_03: log p(C)           (C 가 첫 변수)
    ...
이 분포들이 ordering 마다 얼마나 다른지 비교.

TOP std_med dataset (ordering 분산 큰 것) 과 BOT (작은 것) 두 개 비교.

==============================================================================
전체 플로우
==============================================================================
[1] _results.csv 에서 mean_std_log_score TOP / BOT dataset 선정
[2] 각 dataset 에서:
    build_orders(p) (22개)
    for each ordering:
        prefix = ordering[:position_of_C]
        score = predict_log_density(prefix → C)  또는 marginal (prefix 비면)
    22 분포 step-histogram overlay

==============================================================================
필요한 사전 파일
==============================================================================
./KS_results/_results.csv  (run_ks aggregate 후)

==============================================================================
실행
==============================================================================
python explore_C_in_contexts.py

==============================================================================
결과 / 해석
==============================================================================
KS_results/C_in_contexts.png  (2행: TOP / BOT dataset)

해석:
    TOP dataset 에선 같은 C 라도 prefix 가 바뀌면 분포가 크게 흔들림
    BOT dataset 에선 거의 비슷 → ordering 영향 작음
==============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ks_helpers import (DATA_DIR, RANDOM_SEED, set_global_seed, load_npz,
                         build_split, cap_train, build_orders,
                         predict_log_density, marginal_log_density)

RESULTS_CSV = './KS_results/_results.csv'
TARGET_VAR  = 0       # 추적할 변수 C index (필요시 변경)


def collect_C_scores(X_tr, X_te, orders, C):
    results = []
    for (ord_name, ordering) in orders:
        j = ordering.index(C)
        prefix = tuple(ordering[:j])
        if j == 0:
            score = marginal_log_density(X_tr[:, C], X_te[:, C])
        else:
            score = predict_log_density(X_tr[:, list(prefix)], X_tr[:, C],
                                          X_te[:, list(prefix)], X_te[:, C])
        results.append({'name': ord_name, 'prefix': prefix,
                         'n_prefix': j, 'score': score})
    return results


def plot_axes(ax, results, title):
    vals = np.concatenate([r['score'] for r in results])
    lo, hi = np.quantile(vals, [0.01, 0.99])
    bins = np.linspace(lo, hi, 25)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for r, color in zip(results, colors):
        ax.hist(r['score'], bins=bins, histtype='step', linewidth=1.0,
                color=color, alpha=0.6)
    ax.set_xlabel('log p_hat(C | prefix)'); ax.set_ylabel('count')
    ax.set_title(title)


def run_for_dataset(name, C=TARGET_VAR):
    path = os.path.join(DATA_DIR, f'{name}.npz')
    if not os.path.exists(path):
        print(f'{name}: missing'); return None
    X, y = load_npz(path)
    X_tr, X_te, _ = build_split(X, y)
    X_tr = cap_train(X_tr)
    p = X.shape[1]
    if C >= p:
        C = 0
    orders = build_orders(p)
    print(f'\n[{name}] p={p}, n_tr={len(X_tr)}, n_te={len(X_te)}, '
          f'M={len(orders)}, C=v{C}')
    return name, collect_C_scores(X_tr, X_te, orders, C), C


def main():
    set_global_seed(RANDOM_SEED)
    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values('mean_std_log_score', ascending=False)
    top = df.iloc[0]['dataset']
    bot = df.iloc[-1]['dataset']
    print(f'TOP std: {top}\nBOT std: {bot}')

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for ax, name in zip(axes, [top, bot]):
        out = run_for_dataset(name)
        if out is None:
            continue
        _, results, C = out
        plot_axes(ax, results, f'{name}  C=v{C}')
    plt.tight_layout()
    plt.savefig('./KS_results/C_in_contexts.png', dpi=120)
    plt.close(fig)
    print('saved KS_results/C_in_contexts.png')


if __name__ == '__main__':
    main()
