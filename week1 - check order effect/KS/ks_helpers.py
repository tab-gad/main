"""KS 공통 helper — 다른 KS/*.py 가 import.

==============================================================================
목적
==============================================================================
TabPFN AR chain rule 의 conditional log density 계산 + 분산 metric 도구.
HY 와 동일 setting: seed=42, train cap=1000, 22 ordering, BarDistribution.

==============================================================================
주요 함수
==============================================================================
[Setting]
    DEFAULT_DATASETS         : HY 17 dataset 목록
    set_global_seed(seed)    : numpy/torch/random seed 고정

[Dataset 준비]
    load_npz(path)           : .npz 로드 + 상수 변수 제거
    build_split(X, y)        : normal 70/30 split → train_normal / (test_normal + anomaly)
    cap_train(X_tr)          : train > MAX_TRAIN_SAMPLES 면 sampling
    build_orders(p)          : canonical + reverse + 20 random = 22 ordering

[Density]
    predict_log_density(X_tr, y_tr, X_te, y_te)
        TabPFN BarDistribution log_prob(y_te) → conditional log density (n_test,)
    marginal_log_density(y_tr, y_te)
        TabPFN + 더미 1차원 prefix → marginal log density (j=0 step)
    total_log_score(X_tr, X_te, ordering)
        chain rule 합 log p̂(x). (sorted prefix, target) 캐시 사용.

[Cache]
    reset_cache()            : dataset 단위로 호출 (test set 바뀔 때마다)

==============================================================================
환경 변수 override (시간 단축용)
==============================================================================
    MAX_TRAIN=300 N_RAND_ORDERS=5 python run_ks.py ...
==============================================================================
"""
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

import os as _os
DATA_DIR          = '../data/Classical'
RANDOM_SEED       = 42
TEST_SIZE         = 0.30
# 시간 단축 위해 env var 로 override 가능:
#   FAST mode: N_RAND_ORDERS=5 MAX_TRAIN=300 python run_ks.py ...
MAX_TRAIN_SAMPLES = int(_os.environ.get('MAX_TRAIN', 1000))         # default HY=1000
N_RANDOM_ORDERS   = int(_os.environ.get('N_RAND_ORDERS', 20))       # default HY=20 (22 ordering)
N_ESTIMATORS      = int(_os.environ.get('N_EST', 1))
LOG_PROB_FLOOR    = 1e-10
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[ks_helpers] MAX_TRAIN={MAX_TRAIN_SAMPLES}  N_RAND_ORDERS={N_RANDOM_ORDERS}  '
      f'N_EST={N_ESTIMATORS}', flush=True)


def set_global_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- TabPFN BarDistribution log density ----------
def predict_log_density(X_tr, y_tr, X_te, y_te, seed=RANDOM_SEED):
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
    """HY 방식: TabPFN 으로 marginal 추정.
    prefix 자리에 더미 랜덤 1차원 feature 넣어 fit → TabPFN 이 prefix 무시 → marginal.
    """
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()
    if y_train.std() < 1e-12:
        return np.full(len(y_test), np.log(LOG_PROB_FLOOR))
    rng = np.random.default_rng(seed)
    X_tr_dummy = rng.standard_normal((len(y_train), 1)).astype(np.float32)
    X_te_dummy = rng.standard_normal((len(y_test), 1)).astype(np.float32)
    return predict_log_density(X_tr_dummy, y_train, X_te_dummy, y_test, seed=seed)


# ---------- (sorted prefix, target) 캐싱 ----------
# TabPFN 은 입력 column 순서 무관. ordering 끼리 같은 prefix-set + target 조합이
# 자주 등장 → 한 번 계산 후 재사용 (dataset 단위로 reset).
_COND_CACHE = {}
_MARGINAL_CACHE = {}


def reset_cache():
    _COND_CACHE.clear()
    _MARGINAL_CACHE.clear()


def total_log_score(X_tr, X_te, ordering, seed=RANDOM_SEED):
    """chain rule total log p_hat. (sorted prefix, target) 캐시 사용."""
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


# ---------- Ordering bank (HY: canonical + reverse + n_random) ----------
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
        cnt += 1; seen.add(pi)
        orders.append((f'random_order_{cnt:02d}', list(pi)))
    return orders


# ---------- Dataset split (HY 와 동일) ----------
def load_npz(path):
    d = np.load(path)
    X = d['X'].astype(float); y = d['y'].astype(int).ravel()
    keep = X.var(axis=0) > 1e-12         # 상수 변수 제거
    return X[:, keep], y


def build_split(X, y, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """normal 70/30 → train_normal / test_normal. test = test_normal ∪ anomaly."""
    X_n = X[y == 0]; X_a = X[y == 1]
    X_tr, X_te_n = train_test_split(X_n, test_size=test_size,
                                      random_state=seed, shuffle=True)
    X_te = np.vstack([X_te_n, X_a])
    y_te = np.concatenate([np.zeros(len(X_te_n), dtype=int),
                           np.ones(len(X_a), dtype=int)])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_te))
    return X_tr, X_te[perm], y_te[perm]


def cap_train(X_tr, max_rows=MAX_TRAIN_SAMPLES, seed=RANDOM_SEED):
    if max_rows is None or len(X_tr) <= max_rows:
        return X_tr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=max_rows, replace=False)
    return X_tr[idx]


# ---------- HY DEFAULT_DATASETS ----------
DEFAULT_DATASETS = [
    '12_fault.npz', '43_WDBC.npz', '6_cardio.npz', '7_Cardiotocography.npz',
    '18_Ionosphere.npz', '20_letter.npz', '27_PageBlocks.npz',
    '28_pendigits.npz', '29_Pima.npz', '2_annthyroid.npz', '37_Stamps.npz',
    '38_thyroid.npz', '40_vowels.npz', '41_Waveform.npz', '4_breastw.npz',
    '47_yeast.npz', '46_WPBC.npz',
]
