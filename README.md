# Tabular GAD Experiments

## Overview
- **Tabular Generalist Anomaly Detection (GAD)** 실험을 위한 메인 실행 환경
- 실험은 **ADBench Classical tabular datasets**를 기반으로 수행

## Environment Requirements
- **Python version**: `>= 3.6, < 3.10`

## Directory Structure
- `main/`
  - 메인 실행 스크립트 위치
- `main/dataset/Classical/`
  - ADBench Classical 데이터셋 디렉토리  
  - Source: https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical


## Usage Example
```bash
python gad_1111.py -d 4 -se 1 -m 3
```

- 데이터셋(`-d`) 
```
1 : '1_ALOI', 
2 : '2_annthyroid', 
3 : '3_backdoor', 
4 : '4_breastw', 
5 : '5_campaign', 
6 : '6_cardio',  
7 : '7_Cardiotocography', 
8 : '8_celeba', 
9 : '9_census', 
10 : '10_cover', 
```
- 모델(`-m`)
```
0 : 'mlp', 
1 : 'lr', 
2 : 'iforest',
3 : 'dbscan', 
4 : 'deepsad'
```
- 시드(`-se`)

## Using TabPFN
### Setup Steps

1. 브라우저에서 아래 페이지 접속
https://huggingface.co/Prior-Labs/tabpfn_2_5
2. Hugging Face 계정 로그인
3. “Agree and access” 버튼을 클릭하여 라이선스 약관 동의
4. 터미널에서 다음 명령 실행
```bash
hf auth login
```
5. Hugging Face access token 입력
