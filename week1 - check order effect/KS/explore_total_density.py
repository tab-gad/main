"""explore_total_density.py — ordering 별 log p̂ 분포 overlay 시각화.

==============================================================================
목적
==============================================================================
한 dataset 안에서 22 ordering 각각이 test 점들에 부여한 log p̂ 분포가
얼마나 다른지 시각적으로 확인.

==============================================================================
전체 플로우
==============================================================================
[1] dataset 별로:
    load_npz → build_split + cap_train → build_orders (22개)
    for each ordering: total_log_score → (n_test,) 점수 모음
[2] plot_overlay():
    위:  22 ordering 모두 filled hist overlay (alpha=0.30, density 정규화)
    아래: random_order 중 5개 골라 ordering 별 subplot 분리

==============================================================================
실행
==============================================================================
# 단일
python explore_total_density.py
# 7 shard 병렬
for i in 0 1 2 3 4 5 6; do
    python explore_total_density.py $i 7 > logs/td_${i}.txt 2>&1 &
done; wait

==============================================================================
결과 / 해석
==============================================================================
KS_results/total_density_figs/total_density_<dataset>.png

우측 상단 통계 박스:
    mean across ord : 22 ordering 평균값의 평균
    std of means    : ordering 평균값들의 std (분포 좌우 이동 정도)
    mean of stds    : ordering 별 std 의 평균 (데이터 자체 dispersion)

해석:
    overlay 가 한 곳에 겹쳐 보이면 → ordering robust
    좌우로 흩어져 있으면 → ordering 마다 분포 shift 큼
==============================================================================
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ks_helpers import (DATA_DIR, DEFAULT_DATASETS, RANDOM_SEED,
                         set_global_seed, load_npz, build_split, cap_train,
                         build_orders, total_log_score, reset_cache)

OUT_DIR = './KS_results/total_density_figs'
BIN_N   = 30

# ---------- 병렬 (shard) ----------
_argv = sys.argv[1:]
if len(_argv) >= 2 and _argv[0].isdigit() and _argv[1].isdigit():
    SHARD_ID, N_SHARDS = int(_argv[0]), int(_argv[1])
elif 'SHARD_ID' in os.environ and 'N_SHARDS' in os.environ:
    SHARD_ID, N_SHARDS = int(os.environ['SHARD_ID']), int(os.environ['N_SHARDS'])
else:
    SHARD_ID, N_SHARDS = 0, 1
print(f'[startup] SHARD_ID={SHARD_ID}/{N_SHARDS}', flush=True)


def plot_overlay(scores_by_ord, name, save_path, n_show=5, seed=RANDOM_SEED):
    """위: 22 ordering 모두 overlay (alpha=0.30, density).
    아래: 22개 중 랜덤 n_show 개만 ordering 별 subplot 분리.
    """
    M = len(scores_by_ord)
    keys = list(scores_by_ord.keys())
    all_vals = np.concatenate(list(scores_by_ord.values()))
    lo, hi = np.quantile(all_vals, [0.01, 0.99])
    pad = (hi - lo) * 0.05 + 1e-6
    bins = np.linspace(lo - pad, hi + pad, BIN_N + 1)
    colors_all = plt.cm.viridis(np.linspace(0, 1, M))

    # y 통일 (density)
    max_d = 0.0
    for s in scores_by_ord.values():
        c, _ = np.histogram(s, bins=bins, density=True)
        max_d = max(max_d, float(c.max()))
    y_top = max_d * 1.15 + 1e-6

    # 아래에 보일 random sample (random_order 들 중 n_show)
    random_keys = [k for k in keys if k.startswith('random_order')]
    rng = np.random.default_rng(seed)
    pick_n = min(n_show, len(random_keys))
    picked = list(rng.choice(random_keys, size=pick_n, replace=False))

    fig = plt.figure(figsize=(13, 6.5))
    # 위: 전체 overlay
    ax_top = plt.subplot2grid((2, pick_n), (0, 0), colspan=pick_n)
    for (k, s), color in zip(scores_by_ord.items(), colors_all):
        ax_top.hist(s, bins=bins, density=True, histtype='stepfilled',
                    color=color, alpha=0.30)
    ax_top.set_xlim(bins[0], bins[-1]); ax_top.set_ylim(0, y_top)
    ax_top.set_xlabel('total log p_hat(x)'); ax_top.set_ylabel('density')
    ax_top.set_title(f'{name}: total log p_hat per ordering (M={M}, alpha=0.30)')

    means = np.array([s.mean() for s in scores_by_ord.values()])
    stds  = np.array([s.std()  for s in scores_by_ord.values()])
    info = (f'mean across ord: {means.mean():+.2f}\n'
            f'std of means:    {means.std():.3f}\n'
            f'mean of stds:    {stds.mean():.3f}')
    ax_top.text(0.98, 0.97, info, transform=ax_top.transAxes, ha='right', va='top',
                fontsize=8, family='monospace',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    # 아래: 랜덤 pick_n 개 ordering 별 분리
    picked_colors = plt.cm.viridis(np.linspace(0, 1, pick_n))
    for col_idx, (k, color) in enumerate(zip(picked, picked_colors)):
        ax = plt.subplot2grid((2, pick_n), (1, col_idx))
        ax.hist(scores_by_ord[k], bins=bins, density=True,
                color=color, alpha=0.85)
        ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(0, y_top)
        ax.set_title(k, fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('density')
        else:
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_global_seed(RANDOM_SEED)
    my_files = [f for i, f in enumerate(DEFAULT_DATASETS) if i % N_SHARDS == SHARD_ID]
    print(f'shard {SHARD_ID}/{N_SHARDS}: {len(my_files)} dataset')
    for fname in my_files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f'{fname}: missing'); continue
        name = fname.replace('.npz', '')
        reset_cache()                        # ★ dataset 단위로 cache reset
        X, y = load_npz(path)
        X_tr, X_te, _ = build_split(X, y)
        X_tr = cap_train(X_tr)
        p = X.shape[1]
        orders = build_orders(p)
        print(f'[{name}] p={p}, n_tr={len(X_tr)}, n_te={len(X_te)}, '
              f'M={len(orders)} ordering ...')
        scores_by_ord = {}
        t0 = time.time()
        for ord_name, ordering in orders:
            scores_by_ord[ord_name] = total_log_score(X_tr, X_te, ordering)
            print(f'  {ord_name:20s} mean={scores_by_ord[ord_name].mean():+.2f}',
                  flush=True)
        plot_overlay(scores_by_ord, name,
                      os.path.join(OUT_DIR, f'total_density_{name}.png'))
        print(f'  saved ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
