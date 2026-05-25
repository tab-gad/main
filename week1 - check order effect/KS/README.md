# KS 폴더 — TabPFN AR ordering sensitivity (HY setting 일치)

## 목적
adbench Classical dataset 에서 TabPFN AR pseudo log-density 가 **변수 ordering 에 따라 얼마나 달라지는가** 측정 + 회귀.

## 공통 setting (HY 와 일치)
- seed = 42 (numpy / torch / random 모두 고정)
- train = normal 70%, **최대 1000 cap**
- test = normal 30% + anomaly 전부
- ordering = canonical + reverse + **20 random = 22 ordering**
- density = TabPFN FullSupportBarDistribution.log_prob (Gaussian 가정 X)
- marginal (j=0) = TabPFN + 더미 1차원 prefix
- log_prob_floor = 1e-10
- dataset = HY DEFAULT_DATASETS (17개)
- 변경하려면 env var: `N_RAND_ORDERS=5 MAX_TRAIN=300 python ...`

## 파일별 역할

### 1. `ks_helpers.py` — 공통 helper (다른 파일이 import)
- `set_global_seed(seed)` : seed 고정
- `predict_log_density(X_tr, y_tr, X_te, y_te)` : TabPFN BarDistribution → log p̂(y|x)
- `marginal_log_density(y_tr, y_te)` : 더미 prefix 로 marginal
- `total_log_score(X_tr, X_te, ordering)` : chain rule 합 (+ (prefix-set, target) 캐시)
- `reset_cache()` : dataset 단위 cache 초기화
- `build_orders(p)` : canonical + reverse + 20 random
- `build_split(X, y)`, `cap_train(X_tr)` : HY 식 train/test split
- `DEFAULT_DATASETS` : HY 17 dataset 목록

### 2. `run_ks.py` — 메인 측정 + 회귀 (★ 가장 먼저 돌릴 것)
**무엇**: 17 dataset 각각에 대해 22 ordering 의 chain rule log p̂ 계산 → 분산 metric + 회귀.

**실행**:
```bash
# 7 shard 병렬
for i in 0 1 2 3 4 5 6; do
    python run_ks.py $i 7 > logs/run_${i}.txt 2>&1 &
done; wait
python run_ks.py aggregate
```

**결과 파일** (`KS_results/`):
- `_results_shard{0..6}.csv` → `_results.csv` : dataset × (descriptor + ordering 분산 metric)
- `_sample_scores_shard{0..6}.csv` → `_sample_scores.csv` : dataset × ordering × sample 별 log score
- `_regression.txt` : Spearman + Pearson + Lasso + Ridge 결과

**핵심 컬럼**:
| 컬럼 | 의미 |
|---|---|
| `mean_std_log_score` | 한 sample 의 22 ordering log p̂ std → dataset 평균. **ordering 분산 척도** |
| `p90_std_log_score` | 위 std 의 90 percentile |
| `normal_mean_std` / `anomaly_mean_std` | 정상/이상 점만 |
| `mean_spearman_rho` / `min_spearman_rho` | ordering pair 의 sample ranking 상관 (1=ranking 보존) |
| `p`, `n_normal`, `anom_ratio`, `mean_abs_corr`, `eff_rank_ratio` 등 | dataset descriptor (회귀 feature) |

**해석**:
- `mean_std_log_score` 큼 → ordering 마다 score 절댓값 변동 큼 (예: fault).
- `min_spearman_rho` 0.9+ → ordering 바뀌어도 sample ranking 거의 유지.

### 3. `explore_total_density.py` — ordering 별 분포 overlay
**무엇**: 22 ordering 각각의 test 점 log p̂ 분포를 한 그림에 overlay.

**실행** (병렬 가능):
```bash
for i in 0 1 2 3 4 5 6; do
    python explore_total_density.py $i 7 > logs/td_${i}.txt 2>&1 &
done; wait
```

**결과**: `KS_results/total_density_figs/total_density_<dataset>.png` (dataset 별 1개)

**그림 구성**:
- 위: 22 ordering 전부 filled hist overlay (alpha=0.30, density 정규화)
- 아래: random_order 중 5개만 골라 ordering 별 subplot 분리
- 우측 상단 통계:
  - `mean across ord` : 22 ordering 평균값의 평균
  - `std of means` : ordering 평균값들의 std (분포 좌우 이동 정도)
  - `mean of stds` : ordering 별 std 의 평균 (데이터 자체 dispersion)

**해석**: ordering 끼리 분포가 좌우로 흩어져 있으면 ordering shift 큼.

### 4. `explore_corr_pairs.py` — 변수쌍 chain rule joint 비교
**무엇**: dataset 마다 상관 큰/중간/작은 변수쌍 3개에서 P(A), P(B|A), joint A→B vs B→A 7가지 분포 비교.

**실행** (병렬 가능):
```bash
for i in 0 1 2 3 4 5 6; do
    python explore_corr_pairs.py $i 7 > logs/pairs_${i}.txt 2>&1 &
done; wait
```

**결과**: `KS_results/pairs_figs/pairs_<dataset>.png`

**그림 구성**: 3행 (쌍 high/mid/low) × 7열 (P(A), P(B|A), joint_AB, P(A|B), P(B), joint_BA, joint 차이)
- 차이 column: 양수=red, 음수=navy, 0 기준선.

**해석**: 차이 histogram 이 0 주변에 좁게 모이면 두 ordering joint 가 거의 일치 (ordering robust). 넓거나 치우치면 ordering 영향 큼.

### 5. `explore_C_in_contexts.py` — 한 변수 C 의 prefix 별 평가
**무엇**: 한 변수 C 가 22 ordering 마다 다른 prefix 에서 평가됨 → local log p(C|prefix) 분포 비교.

**필요**: `_results.csv` (run_ks 먼저).

**실행**: `python explore_C_in_contexts.py`
**결과**: `KS_results/C_in_contexts.png` (TOP/BOT std_med dataset 2개 비교).

**해석**: 같은 C 라도 prefix 가 바뀌면 분포가 흔들리는지. 흔들리면 그 변수가 ordering-sensitive.

### 6. `explore_variable_breakdown.py` — TOP/BOT dataset 변수별 분해
**무엇**: ordering variance = sum_v Var[c_v] + cross-cov 로 분해.
- Var[c_v] 가 한 변수에 몰리면 → 그 변수만 흔드는 것
- 골고루 → 여러 변수 누적

**필요**: `_results.csv`.

**실행**: `python explore_variable_breakdown.py`
**결과**: `KS_results/breakdown_compare.png` (TOP/BOT std_med 두 dataset bar chart)

### 7-1. `check_seed_sensitivity.py` — seed 변경 시 mean_std_log_score 변동
**무엇**: cardio / fault 두 dataset 에서 seed 5개 (42, 0, 1, 2, 3) 로 측정 후 mean_std_log_score / spearman_rho 변동 폭 확인.

**용도**: HY vs KS 결과 차이 (~1-2) 가 seed 차이 만으로 설명되는지 진단.

**실행**: `python check_seed_sensitivity.py`
**결과**: `KS_results/seed_sensitivity.csv` + stdout 요약 (dataset 별 seed 간 mean/std/min/max).

**해석**:
- seed 간 std 가 작음 (< 0.1) → 차이는 seed 외 다른 원인 (HY 코드 차이, marginal 정의, 등)
- seed 간 std 가 큼 (~1-2) → seed 변동만으로 차이 설명 가능

### 8. `explore_variable_breakdown_full.py` — 모든 dataset 변수별 분해
**무엇**: 위와 동일하되 17 dataset 모두.

**실행** (병렬 가능):
```bash
for i in 0 1 2 3; do python explore_variable_breakdown_full.py $i 4 & done; wait
```
**결과**: `KS_results/breakdown_full_figs/breakdown_<dataset>.png`

## 전체 실행 순서
```bash
mkdir -p logs

# 1) 메인 측정 (가장 오래 — 1-2 시간)
for i in 0 1 2 3 4 5 6; do
    python run_ks.py $i 7 > logs/run_${i}.txt 2>&1 &
done; wait
python run_ks.py aggregate > logs/run_agg.txt 2>&1

# 2) 시각화 (병렬)
for i in 0 1 2 3 4 5 6; do
    python explore_total_density.py $i 7 > logs/td_${i}.txt 2>&1 &
done; wait

for i in 0 1 2 3 4 5 6; do
    python explore_corr_pairs.py $i 7 > logs/pairs_${i}.txt 2>&1 &
done; wait

for i in 0 1 2 3; do
    python explore_variable_breakdown_full.py $i 4 > logs/vbf_${i}.txt 2>&1 &
done; wait

# 3) TOP/BOT 단일
python explore_C_in_contexts.py > logs/C.txt 2>&1
python explore_variable_breakdown.py > logs/var.txt 2>&1
```

## 결과 해석 한 줄 요약
- **mean_std_log_score** 큰 dataset → ordering 에 따라 log p̂ 절댓값이 흔들림
- **mean_spearman_rho** 1에 가까움 → ordering 바뀌어도 ranking 보존 (AUC 영향 X)
- **회귀 (Lasso/Ridge) R²_cv** → 측정 인자로 ordering 분산을 얼마나 예측하나
- **유의 인자** (Spearman p<0.05) → ordering 분산의 주된 원인 (예: 변수 수 p, eff_rank_ratio)
