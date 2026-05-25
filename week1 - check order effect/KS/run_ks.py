"""run_ks.py — 메인 측정 + 회귀.

==============================================================================
목적
==============================================================================
17 dataset 각각에 대해 22 ordering 의 chain rule log p̂ 계산 →
sample 별 ordering 분산 metric (mean_std_log_score 등) + dataset descriptor 저장.
모든 shard 끝난 후 회귀 (Spearman / Pearson / Lasso / Ridge) 자동 실행.

==============================================================================
전체 플로우
==============================================================================
[1] argparse / env var 로 shard 결정
[2] 이 shard 가 맡을 dataset 만 iter
    for each dataset:
        compute_descriptors(X, y)              # 9 feature
        build_split(X, y) + cap_train          # train/test
        build_orders(p)                        # 22 ordering
        for each ordering:
            total_log_score(X_tr, X_te, π)     # chain rule log p̂
        compute per-sample std, mean_spearman_rho, ...
        append row → _results_shard{ID}.csv
[3] (별도 호출) python run_ks.py aggregate
    → 모든 shard 합쳐 _results.csv + _sample_scores.csv
    → regression() 호출 → _regression.txt

==============================================================================
실행
==============================================================================
# 7 shard 병렬
for i in 0 1 2 3 4 5 6; do
    python run_ks.py $i 7 > logs/run_${i}.txt 2>&1 &
done; wait
python run_ks.py aggregate > logs/run_agg.txt 2>&1

==============================================================================
결과 파일 (KS_results/)
==============================================================================
_results.csv          dataset × (descriptor + 분산 metric)
_sample_scores.csv    dataset × ordering × sample 별 log score
_regression.txt       Spearman/Pearson/Lasso/Ridge 결과

핵심 컬럼:
    mean_std_log_score  : 한 sample 의 22 ordering log p̂ std → dataset 평균 (★ target)
    p90_std_log_score   : 위 std 의 90 percentile
    normal_mean_std / anomaly_mean_std : 정상/이상 점만
    mean_spearman_rho / min_spearman_rho : ordering pair sample ranking 상관

해석:
    mean_std_log_score 큼 → ordering 마다 score 절댓값 흔들림 (예: fault)
    mean_spearman_rho 1 가까움 → ranking 보존 (AUC 영향 X)
==============================================================================
"""
import os
import sys
import random
import time
import math
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor


# ============================================================
# 설정 (HY 와 동일)
# ============================================================
DATA_DIR = '../data/Classical'          # KS/ 에서 한 단계 위
OUT_DIR  = './KS_results'
RANDOM_SEED       = 42
TEST_SIZE         = 0.30
# env var override:  N_RAND_ORDERS=5 MAX_TRAIN=300 python run_ks.py ...
MAX_TRAIN_SAMPLES = int(os.environ.get('MAX_TRAIN', 1000))
N_RANDOM_ORDERS   = int(os.environ.get('N_RAND_ORDERS', 20))
N_ESTIMATORS      = int(os.environ.get('N_EST', 1))
LOG_PROB_FLOOR    = 1e-10
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_DATASETS = [
    '12_fault.npz', '43_WDBC.npz', '6_cardio.npz', '7_Cardiotocography.npz',
    '18_Ionosphere.npz', '20_letter.npz', '27_PageBlocks.npz',
    '28_pendigits.npz', '29_Pima.npz', '2_annthyroid.npz', '37_Stamps.npz',
    '38_thyroid.npz', '40_vowels.npz', '41_Waveform.npz', '4_breastw.npz',
    '47_yeast.npz', '46_WPBC.npz',
]

# ---------- 병렬 (shard) ----------
# 사용 예:
#   python run_ks.py 0 7 > log0.txt 2>&1 &
#   ...
#   python run_ks.py aggregate     # 모든 shard 합쳐 _results.csv
_argv = sys.argv[1:]
if _argv and _argv[0] == 'aggregate':
    SHARD_ID, N_SHARDS, AGGREGATE = 0, 1, True
elif len(_argv) >= 2 and _argv[0].isdigit() and _argv[1].isdigit():
    SHARD_ID, N_SHARDS, AGGREGATE = int(_argv[0]), int(_argv[1]), False
elif 'SHARD_ID' in os.environ and 'N_SHARDS' in os.environ:
    SHARD_ID, N_SHARDS, AGGREGATE = int(os.environ['SHARD_ID']), int(os.environ['N_SHARDS']), False
else:
    SHARD_ID, N_SHARDS, AGGREGATE = 0, 1, False
CSV_PATH = f'./KS_results/_results_shard{SHARD_ID}.csv' if N_SHARDS > 1 else './KS_results/_results.csv'
SAMPLES_PATH = f'./KS_results/_sample_scores_shard{SHARD_ID}.csv' if N_SHARDS > 1 else './KS_results/_sample_scores.csv'
print(f'[startup] device={DEVICE}  SHARD_ID={SHARD_ID}/{N_SHARDS}  agg={AGGREGATE}', flush=True)


# ============================================================
# Seed 고정 (HY: set_global_seed)
# ============================================================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TabPFN density (BarDistribution log_prob, floor 적용)
# ============================================================
def predict_log_density(X_tr, y_tr, X_te, y_te, seed=RANDOM_SEED):
    """conditional log density. floor 적용."""
    reg = TabPFNRegressor(device=DEVICE, n_estimators=N_ESTIMATORS,
                          random_state=seed)
    reg.fit(X_tr, y_tr)
    out = reg.predict(X_te, output_type='full')
    criterion = out['criterion']
    logits = out['logits']
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    y_t = torch.as_tensor(np.asarray(y_te), dtype=logits.dtype, device=logits.device)

    if hasattr(criterion, 'log_prob'):
        try:
            lp = criterion.log_prob(logits, y_t)
            lp = np.asarray(lp.detach().cpu()).ravel()
            return np.maximum(lp, np.log(LOG_PROB_FLOOR))
        except Exception:
            pass

    # borders + softmax fallback
    borders = criterion.borders
    if not isinstance(borders, torch.Tensor):
        borders = torch.as_tensor(borders, dtype=logits.dtype)
    borders = borders.to(logits.device).float()
    K = logits.shape[-1]
    probs = F.softmax(logits.float(), dim=-1)
    widths = (borders[1:] - borders[:-1]).clamp_min(1e-12)
    density = probs / widths
    bin_idx = torch.searchsorted(borders, y_t) - 1
    bin_idx = bin_idx.clamp(0, K - 1)
    sel = density[torch.arange(len(y_t), device=density.device), bin_idx]
    lp = torch.log(sel.clamp_min(LOG_PROB_FLOOR))
    return np.asarray(lp.detach().cpu()).ravel()


def marginal_log_density(y_train, y_test, seed=RANDOM_SEED):
    """HY 방식: TabPFN + 더미 prefix 로 marginal 추정."""
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()
    if y_train.std() < 1e-12:
        return np.full(len(y_test), np.log(LOG_PROB_FLOOR))
    rng = np.random.default_rng(seed)
    X_tr_dummy = rng.standard_normal((len(y_train), 1)).astype(np.float32)
    X_te_dummy = rng.standard_normal((len(y_test), 1)).astype(np.float32)
    return predict_log_density(X_tr_dummy, y_train, X_te_dummy, y_test, seed=seed)


# ---------- (sorted prefix, target) 캐싱 ----------
# 같은 prefix-set + target 조합은 ordering 마다 반복 → 한 번 계산 후 재사용.
# TabPFN 은 column 순서 무관이므로 sorted prefix 키 사용. dataset 단위로 reset.
_COND_CACHE = {}
_MARGINAL_CACHE = {}


def reset_cache():
    _COND_CACHE.clear()
    _MARGINAL_CACHE.clear()


def total_log_score(X_tr, X_te, ordering, seed=RANDOM_SEED):
    """chain rule total log p_hat. (sorted prefix, target) 캐시 사용. shape (n_test,)."""
    n_test = X_te.shape[0]
    log_p = np.zeros(n_test)
    for j, var in enumerate(ordering):
        if j == 0:
            if var not in _MARGINAL_CACHE:
                _MARGINAL_CACHE[var] = marginal_log_density(X_tr[:, var], X_te[:, var])
            log_p += _MARGINAL_CACHE[var]
        else:
            key = (tuple(sorted(ordering[:j])), var)
            if key not in _COND_CACHE:
                prefix = list(key[0])
                _COND_CACHE[key] = predict_log_density(
                    X_tr[:, prefix], X_tr[:, var],
                    X_te[:, prefix], X_te[:, var], seed=seed)
            log_p += _COND_CACHE[key]
    return log_p


# ============================================================
# Ordering bank (HY: canonical + reverse + n_random)
# ============================================================
def build_orders(p, n_random=N_RANDOM_ORDERS, seed=RANDOM_SEED):
    canonical = list(range(p))
    reverse = list(reversed(canonical))
    orders = [('canonical_order', canonical),
              ('reverse_order',   reverse)]
    seen = {tuple(canonical), tuple(reverse)}
    rng = np.random.default_rng(seed)
    cnt = 0
    while cnt < n_random:
        pi = tuple(rng.permutation(p).tolist())
        if pi in seen:
            continue
        cnt += 1
        seen.add(pi)
        orders.append((f'random_order_{cnt:02d}', list(pi)))
    return orders


# ============================================================
# Dataset 처리 (HY build_split 와 동일 로직)
# ============================================================
def build_split(X, y, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """normal: 70/30 split → train_normal / test_normal.
    test = test_normal ∪ anomaly 전부.
    """
    normal_mask = y == 0
    anomaly_mask = y == 1
    X_normal = X[normal_mask]
    X_anomaly = X[anomaly_mask]
    X_tr, X_te_n = train_test_split(X_normal, test_size=test_size,
                                      random_state=seed, shuffle=True)
    X_te = np.vstack([X_te_n, X_anomaly])
    y_te = np.concatenate([np.zeros(len(X_te_n), dtype=int),
                           np.ones(len(X_anomaly), dtype=int)])
    # shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_te))
    return X_tr, X_te[perm], y_te[perm]


def cap_train(X_tr, max_rows=MAX_TRAIN_SAMPLES, seed=RANDOM_SEED):
    """train cap. max_rows=None 이면 전체 사용."""
    if max_rows is None or len(X_tr) <= max_rows:
        return X_tr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=max_rows, replace=False)
    return X_tr[idx]


# ============================================================
# 한 dataset 처리
# ============================================================
def compute_descriptors(X, y):
    """회귀용 dataset 특성."""
    from scipy.stats import skew as _skew, kurtosis as _kurt
    n, p = X.shape
    Xn = X[y == 0] if (y == 0).sum() > 10 else X
    Xn_z = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
    if p >= 2:
        C = np.corrcoef(Xn_z, rowvar=False)
        iu = np.triu_indices_from(C, k=1)
        mean_abs_corr = float(np.abs(C[iu]).mean())
        s = np.linalg.svd(Xn_z, compute_uv=False)
        var = s ** 2
        cum = np.cumsum(var) / var.sum()
        eff_r = int(np.searchsorted(cum, 0.95) + 1)
        pca_first = float(var[0] / var.sum())
    else:
        mean_abs_corr, eff_r, pca_first = 0.0, 1, 1.0
    mean_abs_skew = float(np.mean(np.abs(_skew(Xn_z, axis=0, nan_policy='omit'))))
    mean_abs_kurt = float(np.mean(np.abs(_kurt(Xn_z, axis=0, nan_policy='omit'))))
    raw_var = X.var(axis=0)
    min_var_ratio = float(raw_var.min() / max(raw_var.max(), 1e-12))
    return {
        'p': p, 'n_normal': int((y == 0).sum()),
        'anom_ratio': float((y == 1).mean()),
        'mean_abs_corr': mean_abs_corr,
        'mean_abs_skew': mean_abs_skew,
        'mean_abs_kurt': mean_abs_kurt,
        'eff_rank_ratio': eff_r / p,
        'pca_first_expl': pca_first,
        'min_var_ratio': min_var_ratio,
    }


def run_one(name, X, y):
    set_global_seed(RANDOM_SEED)
    reset_cache()    # dataset 단위로 cache reset
    desc = compute_descriptors(X, y)
    X_tr, X_te, y_te = build_split(X, y)
    X_tr = cap_train(X_tr)
    p = X.shape[1]
    orders = build_orders(p)

    n_normal_te = int((y_te == 0).sum())
    n_anom_te   = int((y_te == 1).sum())
    print(f'[{name}] p={p}, n_train={len(X_tr)}, n_test={len(X_te)} '
          f'(normal {n_normal_te} + anomaly {n_anom_te}), M={len(orders)}')

    scores_by_ord = {}
    for ord_name, ordering in orders:
        t0 = time.time()
        s = total_log_score(X_tr, X_te, ordering)
        scores_by_ord[ord_name] = s
        print(f'  {ord_name:20s} mean={s.mean():+.3f}  std={s.std():.3f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    # per-sample stats across orderings
    S = np.stack([scores_by_ord[k] for k, _ in orders], axis=0)   # (M, n)
    per_sample_std = S.std(axis=0)                                 # (n,)

    # rank correlation (pairs)
    M = len(orders)
    rhos = []
    for i in range(M):
        for j in range(i + 1, M):
            rho, _ = spearmanr(S[i], S[j])
            rhos.append(rho)
    mean_rho = float(np.mean(rhos))
    min_rho  = float(np.min(rhos))

    mask_n = (y_te == 0)
    mask_a = (y_te == 1)
    row = {
        'dataset': name, **desc,
        'train_normal_count_after_cap': len(X_tr), 'n_test': len(X_te),
        'n_test_normal': int(mask_n.sum()), 'n_test_anomaly': int(mask_a.sum()),
        'n_orderings': M,
        'mean_std_log_score':    float(per_sample_std.mean()),
        'p90_std_log_score':     float(np.quantile(per_sample_std, 0.90)),
        'normal_mean_std':       float(per_sample_std[mask_n].mean()) if mask_n.any() else np.nan,
        'anomaly_mean_std':      float(per_sample_std[mask_a].mean()) if mask_a.any() else np.nan,
        'mean_spearman_rho':     mean_rho,
        'min_spearman_rho':      min_rho,
    }

    # sample-level frame (옵션)
    sample_rows = []
    for k, (ord_name, _) in enumerate(orders):
        for i in range(len(X_te)):
            sample_rows.append({
                'dataset': name, 'ordering': ord_name, 'sample_idx': i,
                'y': int(y_te[i]), 'total_log_score': float(S[k, i]),
            })
    return row, sample_rows


# ============================================================
# 메인
# ============================================================
def regression(df):
    """Spearman + Pearson + LassoCV + RidgeCV.
    target = mean_std_log_score, normal_mean_std, anomaly_mean_std.
    """
    from scipy.stats import pearsonr
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.model_selection import cross_val_score, KFold

    ok = df[df['mean_std_log_score'].notna()].copy()
    features = ['p', 'n_normal', 'anom_ratio',
                'mean_abs_corr', 'mean_abs_skew', 'mean_abs_kurt',
                'eff_rank_ratio', 'pca_first_expl', 'min_var_ratio']
    # n_train_per_feature
    if 'train_normal_count_after_cap' in ok.columns:
        ok['n_train_per_feature'] = ok['train_normal_count_after_cap'] / ok['p']
    targets = ['mean_std_log_score', 'normal_mean_std', 'anomaly_mean_std']

    lines = [f'n_datasets (ok) = {len(ok)}',
             'targets = ' + ', '.join(targets),
             f'features = {features}', '']

    for t in targets:
        if t not in ok.columns:
            continue
        lines.append(f'\n========== TARGET: {t} ==========')
        lines.append('\n--- Spearman ρ (단변량) ---')
        rho_rows = []
        for f in features:
            sub = ok.dropna(subset=[f, t])
            if sub[f].nunique() < 2:
                rho_rows.append({'feature': f, 'rho': np.nan, 'pval': np.nan,
                                  'pearson_r': np.nan, 'pearson_p': np.nan})
                continue
            rho, p_s = spearmanr(sub[f], sub[t])
            r, p_p = pearsonr(sub[f], sub[t])
            rho_rows.append({'feature': f, 'rho': rho, 'pval': p_s,
                              'pearson_r': r, 'pearson_p': p_p})
        uni = pd.DataFrame(rho_rows).sort_values('rho', key=lambda s: s.abs(),
                                                   ascending=False)
        lines.append(uni.to_string(index=False))

        # Lasso / Ridge
        df_ok = ok.copy()
        df_ok['log_p'] = np.log(df_ok['p'])
        if 'n_normal' in df_ok.columns:
            df_ok['log_n'] = np.log(df_ok['n_normal'].clip(lower=1))
        X_cols = [c for c in ['log_p', 'log_n', 'anom_ratio',
                               'mean_abs_corr', 'mean_abs_skew', 'mean_abs_kurt',
                               'eff_rank_ratio', 'pca_first_expl', 'min_var_ratio']
                  if c in df_ok.columns]
        Xz = StandardScaler().fit_transform(df_ok[X_cols].fillna(0).values)
        y = df_ok[t].values
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        try:
            lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=cv, max_iter=20000).fit(Xz, y)
            lcv = cross_val_score(LassoCV(alphas=[lasso.alpha_], cv=3, max_iter=20000),
                                    Xz, y, cv=cv, scoring='r2').mean()
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 60), cv=cv).fit(Xz, y)
            rcv = cross_val_score(RidgeCV(alphas=[ridge.alpha_]),
                                    Xz, y, cv=cv, scoring='r2').mean()
            coef_df = pd.DataFrame({
                'feature': X_cols,
                'lasso_coef': lasso.coef_,
                'ridge_coef': ridge.coef_,
            }).reindex(np.argsort(-np.abs(lasso.coef_)))
            lines.append(f'\n--- LassoCV α={lasso.alpha_:.4f}  R²_in={lasso.score(Xz,y):.3f}  '
                          f'R²_cv={lcv:.3f}  nonzero={(lasso.coef_!=0).sum()}/{len(X_cols)} ---')
            lines.append(f'--- RidgeCV α={ridge.alpha_:.4f}  R²_in={ridge.score(Xz,y):.3f}  '
                          f'R²_cv={rcv:.3f} ---')
            lines.append('--- Coef (|lasso| 정렬) ---')
            lines.append(coef_df.to_string(index=False))
        except Exception as e:
            lines.append(f'Lasso/Ridge fail: {e}')

    out = os.path.join(OUT_DIR, '_regression.txt')
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'saved: {out}')


def aggregate():
    """모든 shard CSV 합쳐 _results.csv, _sample_scores.csv 만듦 + 회귀."""
    import glob
    os.makedirs(OUT_DIR, exist_ok=True)
    res_files = sorted(glob.glob(os.path.join(OUT_DIR, '_results_shard*.csv')))
    if res_files:
        df = pd.concat([pd.read_csv(f) for f in res_files], ignore_index=True)
        df = df.drop_duplicates(subset=['dataset'])
        df.to_csv(os.path.join(OUT_DIR, '_results.csv'), index=False)
        print(f'aggregated {len(res_files)} shards → _results.csv ({len(df)} rows)')
        # 회귀
        try:
            regression(df)
        except Exception as e:
            print(f'regression fail: {e}')
    samp_files = sorted(glob.glob(os.path.join(OUT_DIR, '_sample_scores_shard*.csv')))
    dfs = []
    for f in samp_files:
        if os.path.getsize(f) == 0:
            print(f'skip empty: {f}'); continue
        try:
            dfs.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f'skip empty (no header): {f}'); continue
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(os.path.join(OUT_DIR, '_sample_scores.csv'), index=False)
        print(f'aggregated → _sample_scores.csv ({len(df)} rows)')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if AGGREGATE:
        aggregate(); return
    set_global_seed(RANDOM_SEED)

    # 이 shard 가 맡을 dataset
    my_files = [f for i, f in enumerate(DEFAULT_DATASETS) if i % N_SHARDS == SHARD_ID]
    print(f'shard {SHARD_ID}/{N_SHARDS}: {len(my_files)} dataset')

    rows = []
    all_sample_rows = []
    t_start = time.time()
    for fname in my_files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f'{fname}: missing'); continue
        d = np.load(path)
        X = d['X'].astype(float); y = d['y'].astype(int).ravel()
        keep = X.var(axis=0) > 1e-12
        X = X[:, keep]
        try:
            row, samples = run_one(fname.replace('.npz', ''), X, y)
            rows.append(row); all_sample_rows.extend(samples)
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
        except Exception as e:
            print(f'{fname}: FAIL {type(e).__name__}: {e}')

    pd.DataFrame(all_sample_rows).to_csv(SAMPLES_PATH, index=False)
    print(f'\nDONE in {(time.time()-t_start)/60:.1f} min')
    print(f'saved: {CSV_PATH}, {SAMPLES_PATH}')


if __name__ == '__main__':
    main()
