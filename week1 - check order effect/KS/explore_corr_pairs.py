"""explore_corr_pairs.py — 변수쌍 (A, B) 의 chain rule joint 가 ordering 에 따라 다른지.

==============================================================================
목적
==============================================================================
한 dataset 에서 변수 2개만 골라 joint p̂(A, B) 를 ordering A→B 와 B→A
두 가지로 계산해 분포 비교. ordering 효과의 가장 단순한 case 시각화.

==============================================================================
전체 플로우
==============================================================================
[1] dataset 별로:
    load + split + cap → correlation matrix C 계산
    pick_three_pairs(C) : |ρ| 큰/중간/작은 3 쌍 (i, j)
    for each 쌍:
        compute_seven_distributions(X_tr, X_te, i, j)
            log p(A), log p(B|A), joint A→B
            log p(A|B), log p(B), joint B→A
            joint_AB - joint_BA (점별 차이)
    plot_dataset() : 3행 × 7열 histogram

==============================================================================
실행
==============================================================================
# 7 shard 병렬
for i in 0 1 2 3 4 5 6; do
    python explore_corr_pairs.py $i 7 > logs/pairs_${i}.txt 2>&1 &
done; wait

==============================================================================
결과 / 해석
==============================================================================
KS_results/pairs_figs/pairs_<dataset>.png

3 행 = 변수쌍 high/mid/low |ρ|
7 열 = P(A), P(B|A), joint_AB, P(A|B), P(B), joint_BA, joint 차이
차이 column: 양수=red, 음수=navy, 0 기준선

해석:
    joint 차이 hist 가 0 주변에 좁게 모이면 → 두 ordering 일치 (robust)
    넓거나 한쪽으로 치우치면 → ordering 효과 큼
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
                         predict_log_density, marginal_log_density)

OUT_DIR = './KS_results/pairs_figs'
BIN_N   = 20
COL_KEYS = ['P(A)', 'P(B|A)', 'P(A)+P(B|A)',
            'P(A|B)', 'P(B)', 'P(B)+P(A|B)',
            'joint_AB - joint_BA']

# ---------- 병렬 (shard) ----------
_argv = sys.argv[1:]
if len(_argv) >= 2 and _argv[0].isdigit() and _argv[1].isdigit():
    SHARD_ID, N_SHARDS = int(_argv[0]), int(_argv[1])
elif 'SHARD_ID' in os.environ and 'N_SHARDS' in os.environ:
    SHARD_ID, N_SHARDS = int(os.environ['SHARD_ID']), int(os.environ['N_SHARDS'])
else:
    SHARD_ID, N_SHARDS = 0, 1
print(f'[startup] SHARD_ID={SHARD_ID}/{N_SHARDS}', flush=True)


def pick_three_pairs(C):
    p = C.shape[0]
    pairs = [(i, j, float(abs(C[i, j])))
             for i in range(p) for j in range(i + 1, p)]
    pairs.sort(key=lambda x: x[2])
    return {'low': pairs[0], 'mid': pairs[len(pairs)//2], 'high': pairs[-1]}


def compute_seven(X_tr, X_te, i, j):
    A_tr, A_te = X_tr[:, i], X_te[:, i]
    B_tr, B_te = X_tr[:, j], X_te[:, j]
    if A_tr.std() < 1e-9 or B_tr.std() < 1e-9:
        return None
    log_pA = marginal_log_density(A_tr, A_te)
    log_pB = marginal_log_density(B_tr, B_te)
    try:
        log_pB_A = predict_log_density(A_tr[:, None], B_tr, A_te[:, None], B_te)
        log_pA_B = predict_log_density(B_tr[:, None], A_tr, B_te[:, None], A_te)
    except Exception as e:
        print(f'    fail pair ({i},{j}): {e}')
        return None
    joint_AB = log_pA + log_pB_A
    joint_BA = log_pB + log_pA_B
    return {
        'P(A)': log_pA, 'P(B|A)': log_pB_A, 'P(A)+P(B|A)': joint_AB,
        'P(A|B)': log_pA_B, 'P(B)': log_pB, 'P(B)+P(A|B)': joint_BA,
        'joint_AB - joint_BA': joint_AB - joint_BA,
    }


def compute_ranges(rows):
    col_xlim, col_bins, col_ylim = {}, {}, {}
    alpha = 0.05
    for key in COL_KEYS:
        vals = np.concatenate([r[3][key] for _, r in rows])
        lo, hi = float(vals.min()), float(vals.max())
        if key == 'joint_AB - joint_BA':
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        pad = max(hi - lo, 1e-9) * alpha
        col_xlim[key] = (lo - pad, hi + pad)
        col_bins[key] = np.linspace(lo - pad, hi + pad, BIN_N + 1)
        max_c = 0
        for _, r in rows:
            c, _ = np.histogram(r[3][key], bins=col_bins[key])
            max_c = max(max_c, int(c.max()))
        col_ylim[key] = int(max_c * 1.1) + 1
    return col_xlim, col_bins, col_ylim


def plot_dataset(name, mac, pairs_dict, save_path):
    rows = [('high', pairs_dict['high']), ('mid', pairs_dict['mid']),
            ('low', pairs_dict['low'])]
    col_xlim, col_bins, col_ylim = compute_ranges(rows)
    fig, axes = plt.subplots(3, 7, figsize=(20, 9))
    for r_idx, (tag, (i, j, corr_val, dists)) in enumerate(rows):
        for c_idx, key in enumerate(COL_KEYS):
            ax = axes[r_idx, c_idx]
            data = dists[key]
            bins = col_bins[key]
            if key == 'joint_AB - joint_BA':
                neg = data[data < 0]; pos = data[data >= 0]
                ax.hist(neg, bins=bins, color='navy', alpha=0.75)
                ax.hist(pos, bins=bins, color='red',  alpha=0.75)
                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                title = f'{key}\nstd={data.std():.3f}'
            else:
                ax.hist(data, bins=bins, color='C0', alpha=0.7)
                title = key
            ax.set_xlim(col_xlim[key]); ax.set_ylim(0, col_ylim[key])
            if r_idx == 0:
                ax.set_title(title, fontsize=10)
            if c_idx == 0:
                ax.set_ylabel(f'{tag}: ({i},{j})\n|rho|={corr_val:.2f}', fontsize=9)
    fig.suptitle(f'{name}    mean|corr|={mac:.3f}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close(fig)


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
        X, y = load_npz(path)
        if X.shape[1] < 2:
            print(f'{name}: p<2 skip'); continue
        X_tr, X_te, _ = build_split(X, y)
        X_tr = cap_train(X_tr)

        # correlation 으로 쌍 선정 (train normal 만 사용)
        Xn_z = (X_tr - X_tr.mean(0)) / (X_tr.std(0) + 1e-9)
        C = np.corrcoef(Xn_z, rowvar=False)
        iu = np.triu_indices_from(C, k=1)
        mac = float(np.abs(C[iu]).mean())
        pairs = pick_three_pairs(C)

        print(f'\n[{name}] p={X.shape[1]}, n_tr={len(X_tr)}, mac={mac:.3f}')
        per_pair = {}
        skipped = False
        for tag, (i, j, c) in pairs.items():
            print(f'  {tag} pair ({i},{j}) |rho|={c:.3f} ...')
            dists = compute_seven(X_tr, X_te, i, j)
            if dists is None:
                skipped = True; break
            per_pair[tag] = (i, j, c, dists)
        if not skipped:
            plot_dataset(name, mac, per_pair,
                          os.path.join(OUT_DIR, f'pairs_{name}.png'))


if __name__ == '__main__':
    main()
