# ResAD for Tabular Data 📊

A comprehensive implementation of ResAD framework adapted for tabular anomaly detection using TabPFN embeddings.

## 🎯 Overview

This project adapts the ResAD framework for tabular data by combining three key components:

1. ** Pre-trained Feature Extractor**: TabPFN for generating rich embeddings
2. ** Feature Constraintor**: Learnable transformation to enhance anomaly-relevant features  
3. ** Normal Distribution Estimator**: Estimates normal data distribution for anomaly scoring

## Validation

- 12/16: data cross domain - tabpfn embedding까지 확인

## 🏗️ Architecture

```
Tabular Data → TabPFN Extractor → Feature Constraintor → Distribution Estimator → Anomaly Scores
     ↓              ↓                     ↓                      ↓
   Raw Features   Rich Embeddings    Constrained Embeddings   Likelihood Scores
``


### Key Components

### 1. Pre-trained Feature Extractor (TabPFN)
- **역할**: 원본 tabular 데이터를 embedding으로 변환
- **구성**
  - `TabPFNFeatureExtractor`: 단일 설정 기반 embedding
  - `MultiScaleTabPFNExtractor`: 여러 설정을 이용한 multi-scale embedding
- **특징**
  - 사전학습 TabPFN 활용
  - supervised / unsupervised 설정 모두 지원

---

### 2. Feature Constraintor
- **역할**: feature 변동성을 줄여 정상 패턴의 공통 구조를 강조
- **구성**
  - `TabularFeatureConstraintor`: MLP 기반 제약
  - `ResidualConstraintor`: ResAD 스타일 residual 변환
  - `AdaptiveConstraintor`: 입력에 따라 제약 강도 조절
  - `MultiScaleConstraintor`: multi-scale embedding 통합
- **의미**
  - “variation에서 invariance를 추출”하기 위한 핵심 단계

---

### 3. Normal Distribution Estimator
- **역할**: 정상 데이터 분포를 학습하고 이상 점수 계산
- **구성**
  - `NormalDistributionEstimator`: Gaussian + Mahalanobis distance
  - `FlowBasedEstimator`: normalizing flow 기반 분포 추정
  - `EnsembleEstimator`: 여러 추정기의 결합
- **출력**
  - anomaly score (likelihood 또는 distance 기반)


## 📁 Project Structure

```
resad_table/
├── models/                          # Core model components
│   ├── tabpfn_extractor.py         # TabPFN-based feature extraction
│   ├── constraintor.py             # Feature constraint modules
│   └── estimator.py                # Distribution estimation modules
├── datasets/                       # Data loading and preprocessing
│   └── tabular_loader.py           # ADBench and custom dataset support
├── resad_tabular.py                # Main ResAD implementation
├── run_all.py                      # Complete pipeline execution
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install tabpfn-extensions
```

### Basic Usage

```python
# Simple single-dataset experiment
python resad_tabular.py --dataset breastw --mode single_domain

# Cross-domain experiment  
python resad_tabular.py --mode cross_domain --source_datasets breastw pima --target_dataset wdbc

# Complete pipeline with all experiments
python run_all.py --experiment all
```

### Advanced Configuration

```python
# Custom ResAD configuration
python resad_tabular.py \
    --dataset wdbc \
    --use_multiscale \
    --constraintor_type residual \
    --estimator_type flow \
    --epochs 100 \
    --batch_size 128 \
    --lr 5e-4
```

## 📊 Supported Datasets

The framework supports ADBench datasets and custom tabular datasets:

- **ADBench**: breastw, pima, wdbc, wine, cardio, glass, hepatitis, cover
- **Custom**: Any CSV/Parquet file with features and `is_anomaly` column
- **NPZ**: Standard ADBench format with X and y arrays

## 🧪 Experiments

### 1. Single-Domain Evaluation
```bash
python run_all.py --experiment single_domain
```
Evaluates ResAD on individual datasets with train/test splits.

### 2. Cross-Domain Evaluation  
```bash
python run_all.py --experiment cross_domain
```
Tests generalization by training on source datasets and evaluating on target datasets.

### 3. Ablation Studies
```bash
python run_all.py --experiment ablation
```
Analyzes contribution of different components:
- Multi-scale vs single-scale extraction
- Different constraintor types
- Various distribution estimators

### 4. Parameter Sensitivity
```bash
python run_all.py --experiment sensitivity  
```
Studies impact of hyperparameters:
- Learning rate, batch size, epochs
- Regularization weights
- TabPFN configurations

## 🔧 Configuration Options

### Feature Extractor Options
```python
--use_multiscale          # Enable multi-scale TabPFN extraction
--n_scales 3              # Number of scales for multi-scale
--n_estimators 1          # Number of TabPFN estimators  
--n_fold 5                # Cross-validation folds for TabPFN
--use_scaler              # Apply feature scaling
```

### Constraintor Options
```python
--constraintor_type residual    # Type: basic|residual|adaptive
--num_residual_blocks 3         # Number of residual blocks
--dropout_rate 0.1              # Dropout rate for regularization
```

### Estimator Options
```python
--estimator_type normal         # Type: normal|flow|ensemble
--num_flows 4                   # Number of normalizing flow layers
--num_ensemble_estimators 3     # Ensemble size
```

### Training Options
```python
--epochs 50                     # Training epochs
--batch_size 64                 # Batch size
--lr 1e-3                       # Learning rate
--reg_weight 1e-4               # L2 regularization weight
```

## 📈 Results and Analysis

### Output Files

After running experiments, results are saved in timestamped directories:

```
results/run_YYYYMMDD_HHMMSS/
├── single_domain_results.csv          # Single-domain performance
├── cross_domain_results.csv           # Cross-domain performance  
├── ablation_results.csv               # Component ablation
├── sensitivity_results.json           # Parameter sensitivity
├── complete_results.json              # All detailed results
└── *.png                              # Visualization plots
```

### Performance Metrics

- **AUC (Area Under ROC Curve)**: Overall discrimination capability
- **AP (Average Precision)**: Performance on imbalanced data
- **Execution Time**: Computational efficiency

### Visualization

The pipeline automatically generates:
- Performance comparison plots
- Cross-domain transfer analysis
- Ablation study visualizations  
- Parameter sensitivity curves

## 🔬 Technical Details

### TabPFN Integration

TabPFN provides powerful pre-trained representations for tabular data:

```python
# Extract embeddings with TabPFN
embedder = TabPFNEmbedding(tabpfn_clf=TabPFNClassifier(), n_fold=5)
embeddings = embedder.get_embeddings(X_train, y_train, X_test, data_source="test")
```

Key benefits:
- Pre-trained on diverse synthetic tabular data
- Handles mixed data types naturally
- Provides rich semantic embeddings

### Constraint Learning

Feature constraintors learn to emphasize anomaly-relevant patterns:

```python
# Residual constraint learning
class ResidualConstraintor(nn.Module):
    def forward(self, embeddings):
        residual = self.residual_blocks(embeddings)
        return embeddings + residual  # Residual connection
```

Loss function combines:
- Information preservation (cosine similarity)
- Transformation magnitude
- Compactness for normal samples

### Distribution Estimation

Multiple approaches for modeling normal data distribution:

```python
# Gaussian assumption
scores = -log_likelihood_normal(embeddings, mean, covariance)

# Normalizing flows
z, log_det = normalizing_flow(embeddings)  
scores = -log_likelihood_base(z) - log_det
```

## 🛠️ Customization

### Adding New Datasets

```python
# Custom dataset class
class CustomDataset(TabularAnomalyDataset):
    def _load_data(self):
        # Implement custom loading logic
        self.X = load_features()
        self.y = load_labels()
        self.feature_names = get_feature_names()
```

### Custom Constraintors

```python
class MyConstraintor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.transform = nn.Sequential(...)
    
    def forward(self, embeddings):
        return self.transform(embeddings)
```

### Custom Estimators

```python  
class MyEstimator(nn.Module):
    def fit(self, embeddings):
        # Learn distribution from normal embeddings
        pass
    
    def forward(self, embeddings):
        # Return anomaly scores
        return scores
```

## 📊 Example Results

Typical performance on ADBench datasets:

| Dataset | Single-Domain AUC | Cross-Domain AUC |
|---------|-------------------|------------------|
| breastw | 0.0 ± 0.0    | 0.0 ± 0.0    |


### Ablation Study Results

| Component | AUC Impact |
|-----------|------------|
| Baseline | 0.0      |
| + Multi-scale | +0.0 |
| + Residual Constraintor | +0.0 |
| + Flow Estimator | +0.0 |



## 📚 References

- **ResAD Paper**: [ResAD: A Simple Framework for Class Generalizable Anomaly Detection](https://arxiv.org/abs/2410.20047)
- **TabPFN Paper**: [TabPFN: A Transformer that Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)
- **ADBench**: [ADBench: Anomaly Detection Benchmark](https://github.com/Minqi824/ADBench)
