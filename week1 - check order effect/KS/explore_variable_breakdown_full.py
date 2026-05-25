"""explore_variable_breakdown_full.py — 17 dataset 모두 변수별 분해.

==============================================================================
목적
==============================================================================
explore_variable_breakdown 과 같은 분해 (Var_pi[c_v]) 를 모든 dataset 에 적용.
dataset 마다 어떤 변수가 ordering 분산을 만드는지 한눈에.

==============================================================================
필요한 사전 파일
==============================================================================
./KS_results/_results.csv  (std_med 내림차순 정렬에 사용)

==============================================================================
실행
==============================================================================
# 단일
python explore_variable_breakdown_full.py
# 4 shard 병렬
for i in 0 1 2 3; do
    python explore_variable_breakdown_full.py $i 4 > logs/vbf_${i}.txt 2>&1 &
done; wait

==============================================================================
결과 / 해석
==============================================================================
./KS_results/breakdown_full_figs/breakdown_<dataset>.png  (dataset 별 bar chart)

bar = 변수별 Var_pi[c_v], 큰 순서로 정렬.
    첫 막대만 우뚝   → 한 feature 가 ordering 분산 dominant
    평탄           → 여러 feature 가 누적
==============================================================================
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ks_helpers import (DATA_DIR, DEFAULT_DATASETS, RANDOM_SEED,
                         set_global_seed, load_npz, build_split, cap_train,
                         build_orders, predict_log_density, marginal_log_density)

OUT_DIR = './KS_results/breakdown_full_figs'

# shard
_argv = sys.argv[1:]
if len(_argv) >= 2 and _argv[0].isdigit() and _argv[1].isdigit():
    SHARD_ID, N_SHARDS = int(_argv[0]), int(_argv[1])
elif 'SHARD_ID' in os.environ and 'N_SHARDS' in os.environ:
    SHARD_ID, N_SHARDS = int(os.environ['SHARD_ID']), int(os.environ['N_SHARDS'])
else:
    SHARD_ID, N_SHARDS = 0, 1
print(f'[startup] SHARD_ID={SHARD_ID}/{N_SHARDS}', flush=True)


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
        print(f'    ordering {pi_idx+1}/{len(orders)}', flush=True)
    return contribs


def plot_bar(per_var, name, save_path):
    sv = sorted(per_var, key=lambda v: per_var[v], reverse=True)
    vals = [per_var[v] for v in sv]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(vals)), vals, color='C0')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([f'v{v}' for v in sv], rotation=45, fontsize=7)
    ax.set_ylabel('Var_pi[c_v]  (median over samples)')
    ax.set_title(name, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close(fig)


def run_one(name):
    path = os.path.join(DATA_DIR, f'{name}.npz')
    if not os.path.exists(path):
        print(f'{name}: missing'); return
    X, y = load_npz(path)
    X_tr, X_te, _ = build_split(X, y)
    X_tr = cap_train(X_tr)
    orders = build_orders(X.shape[1])
    print(f'\n[{name}] p={X.shape[1]} n_tr={len(X_tr)} n_te={len(X_te)} M={len(orders)}')
    contribs = variable_contributions(X_tr, X_te, orders)
    p = len(contribs)
    per_var = {v: float(np.nanmedian(np.stack(contribs[v], axis=0).var(axis=0, ddof=1)))
               for v in range(p)}
    plot_bar(per_var, name, os.path.join(OUT_DIR, f'breakdown_{name}.png'))
    print(f'  saved breakdown_{name}.png')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_global_seed(RANDOM_SEED)
    names = [f.replace('.npz', '') for f in DEFAULT_DATASETS]
    my_idx = [i for i in range(len(names)) if i % N_SHARDS == SHARD_ID]
    print(f'shard {SHARD_ID}/{N_SHARDS}: {len(my_idx)} dataset')
    for i in my_idx:
        run_one(names[i])


if __name__ == '__main__':
    main()
