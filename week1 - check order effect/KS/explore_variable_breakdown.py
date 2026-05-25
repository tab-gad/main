"""explore_variable_breakdown.py — ordering variance 의 변수별 분해 (TOP / BOT).

==============================================================================
목적
==============================================================================
ordering variance 가
    (a) 특정 한 feature 의 local 흔들림 인지
    (b) 여러 feature 가 조금씩 누적된 결과인지
변수 단위로 분해. TOP / BOT std_med 두 dataset 비교.

분해식:
    Var_pi[sum_v c_v] = sum_v Var_pi[c_v] + 2 sum_{v<w} Cov_pi[c_v, c_w]

==============================================================================
필요한 사전 파일
==============================================================================
./KS_results/_results.csv

==============================================================================
실행 / 결과
==============================================================================
python explore_variable_breakdown.py
→ ./KS_results/breakdown_compare.png

해석:
    bar 첫 막대만 큼  → (a) 한 변수 dominant
    bar 평탄         → (b) 여러 변수 누적
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


def variable_contributions(X_tr, X_te, orders):
    p = X_tr.shape[1]
    contribs = {v: [None] * len(orders) for v in range(p)}
    for pi_idx, (_, ordering) in enumerate(orders):
        for j, var in enumerate(ordering):
            if j == 0:
                ld = marginal_log_density(X_tr[:, var], X_te[:, var])
            else:
                prefix = list(ordering[:j])
                try:
                    ld = predict_log_density(X_tr[:, prefix], X_tr[:, var],
                                              X_te[:, prefix], X_te[:, var])
                except Exception as e:
                    print(f'  fail v={var}: {e}')
                    ld = np.full(X_te.shape[0], np.nan)
            contribs[var][pi_idx] = ld
        print(f'  ordering {pi_idx+1}/{len(orders)} done', flush=True)
    return contribs


def decompose(contribs):
    p = len(contribs)
    C = {v: np.stack(contribs[v], axis=0) for v in range(p)}
    per_var = {v: float(np.nanmedian(C[v].var(axis=0, ddof=1))) for v in range(p)}
    joint = sum(C[v] for v in range(p))
    return {
        'per_var': per_var,
        'sum_pv':  float(sum(per_var.values())),
        'joint':   float(np.nanmedian(joint.var(axis=0, ddof=1))),
    }


def plot_bar(ax, name, std_med, dec):
    per_var = dec['per_var']
    sv = sorted(per_var, key=lambda v: per_var[v], reverse=True)
    vals = [per_var[v] for v in sv]
    ax.bar(range(len(vals)), vals, color='C0')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([f'v{v}' for v in sv], rotation=45, fontsize=7)
    ax.set_ylabel('Var_pi[c_v]  (median over samples)')
    ax.set_title(f'{name}  (std_med={std_med:.3f})')


def run_one(name, std_med, ax):
    path = os.path.join(DATA_DIR, f'{name}.npz')
    X, y = load_npz(path)
    X_tr, X_te, _ = build_split(X, y)
    X_tr = cap_train(X_tr)
    orders = build_orders(X.shape[1])
    print(f'\n[{name}] p={X.shape[1]} n_tr={len(X_tr)} n_te={len(X_te)} M={len(orders)}')
    contribs = variable_contributions(X_tr, X_te, orders)
    dec = decompose(contribs)
    plot_bar(ax, name, std_med, dec)


def main():
    set_global_seed(RANDOM_SEED)
    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values('mean_std_log_score', ascending=False)
    top = df.iloc[0]; bot = df.iloc[-1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    run_one(top['dataset'], float(top['mean_std_log_score']), axes[0])
    run_one(bot['dataset'], float(bot['mean_std_log_score']), axes[1])
    plt.tight_layout()
    plt.savefig('./KS_results/breakdown_compare.png', dpi=120)
    plt.close(fig)
    print('saved KS_results/breakdown_compare.png')


if __name__ == '__main__':
    main()
