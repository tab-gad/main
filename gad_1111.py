import argparse
import os
import torch
import numpy as np
import pandas as pd
import utils.dataset_load as dl
import pdb
import random
from model.mlp import MLPClassifier
from sklearn.linear_model import LogisticRegression
import utils.Function as Ft
from model.relational import make_correlation_features
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


def numbering(dataset, model, sampling, number_name, seed, loss_name):
    data_list = {1 : '1_ALOI', 
                 2 : '2_annthyroid', 
                 3 : '3_backdoor', 
                 4 : '4_breastw', 
                 5 : '5_campaign', 
                 6 : '6_cardio',  
                 7 : '7_Cardiotocography', 
                 8 : '8_celeba', 
                 9 : '9_census', 
                 10 : '10_cover', 
                 11 : '11_donors',
                 12 : '12_fault',
                 13 : '13_fraud',
                 14 : '14_glass',
                 15 : '15_Hepatitis',
                 16 : '16_http',
                 17 : '17_InternetAds',
                 18 : '18_lonosphere',
                 19 : '19_landsat',
                 20 : '20_letter',
                 21 : '21_Lymphography',
                 22 : '22_magic.gamma',
                 23 : '23_mammography',
                 24 : '24_mnist',
                 25 : '25_musk',
                 26 : '26_optdigits',
                 27 : '27_PageBlocks',
                 28 : '28_pendigits',
                 29 : '29_Pima',
                 30 : '30_satellite',
                 31 : '31_satimage-2',
                 32 : '32_shuttle',
                 33 : '33_skin',
                 34 : '34_smtp',
                 35 : '35_SpamBase',
                 36 : '36_speech',
                 37 : '37_Stamps',
                 38 : '38_thyroid',
                 39 : '39_vertebral',
                 40 : '40_vowels',
                 41 : '41_Waveform',
                 42 : '42_WBC',
                 43 : '43_WDBC',
                 44 : '44_Wilt',
                 45 : '45_wine',
                 46 : '46_WPBC',
                 47 : '47_yeast',
                 50 : '50_samsung',
                 51 : '51_samsung_ctabgan',
                 52 : '52_samsung_ctgan' }
    dataset = data_list[dataset]

    model_list = {0 : 'mlp', 
                  1 : 'lr', 
                  2 : 'iforest',
                  3 : 'dbscan', 
                  4 : 'deepsad', 
                  5 : 'devnet', 
                  6 : 'feawad', 
                  7 : 'prenet' ,
                  8 : 'repen'}
    model_name = model_list[model]

    sampling_list = {0 : 'none', 
                     1 : 'smote', 
                     2 : 'borderline-smote',
                     3 : 'adasyn', 
                     4 : 'over-random', 
                     5 : 'tomeklinks', 
                     6 : 'enn',
                     7 : 'down-random' }
    sampling_name = sampling_list[sampling]

    number_list = {0: 'none', 1: 1, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 15, 9: 20, 10: 40, 11: 100}
    number_name = number_list[number_name]

    seed_list = {
        0: [1,11,21,31,41,51,61,71,81,91],
        1: [1],
        2: [0,5,10,15,20,25,30,35,40]
    }
    seed = seed_list[seed]

    loss_list = {0 : 'mse', 1 : 'mfe', 2 : 'msfe', 3 : 'focal', 4 : 'class-balanced' }
    loss_name = loss_list[loss_name]

    return dataset, model_name, sampling_name, number_name, seed, loss_name


def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-d','--data_name', default=0, type=int)
    parser.add_argument('-m','--model', default=0, type=int)
    parser.add_argument('-s','--sampling', default=0, type=int)
    parser.add_argument('-l','--loss', default=0, type=int)
    parser.add_argument('-mi','--minor_number', default=0, type=int)
    parser.add_argument('-se','--seed', default=0, type=int)
    parser.add_argument('-g','--gpu', default=0, type=int)
    parser.add_argument('-e','--epochs', default=150, type=int)
    parser.add_argument('-b','--batch_size', default=128, type=int)
    parser.add_argument('-t','--test', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-ga','--gamma', default=10, type=float)
    parser.add_argument('-al','--alpha', default=0.01, type=float)
    parser.add_argument('-be','--beta', default=0.5, type=float)
    parser.add_argument('-l2','--cbloss', default='squared', type=str)
    parser.add_argument('--file_path', default='./Classical/', type=str)
    parser.add_argument('--model_path', default='./log/model_parameter/', type=str)
    parser.add_argument('--save_path', default='../save_result_lr/', type=str)

    # NEW: 관계특징 옵션 + 교차도메인 평가
    parser.add_argument('--no_relation', action='store_true', help='관계특징 비활성화')
    parser.add_argument('--n_quantiles', default=1000, type=int)
    parser.add_argument('--eval_suites', default='', type=str,
                        help='쉼표로 구분된 타겟 데이터셋 index들. 예: "28,29,31"')

    return parser.parse_known_args()

def model_predict(model_name, model, test_data, trained_feature_dim=None):
    # ✅ 테스트 및 평가
    if model_name == 'iforest' or model_name == 'dbscan':
        test_X = test_data

        # test feature 수 맞추기 (IsolationForest와 동일)
        if test_X.shape[1] > trained_feature_dim:
            test_X = test_X[:, :trained_feature_dim]
            print(f"[WARN] test feature {test_data.shape[1]} → {trained_feature_dim}개로 맞춤")
        elif test_X.shape[1] < trained_feature_dim:
            pad = np.zeros((test_X.shape[0], trained_feature_dim - test_X.shape[1]))
            test_X = np.hstack([test_X, pad])
            print(f"[WARN] test feature {test_data.shape[1]} → {trained_feature_dim}개로 padding")

        if model_name == 'iforest':
            y_pred = model.predict(test_X)
            y_pred = np.where(y_pred == -1, 1, 0)
        elif model_name == 'dbscan':
            y_pred = model.fit_predict(test_X)
            y_pred = np.where(y_pred == -1, 1, 0)

    else:
        y_pred = model.predict(test_data)

    return y_pred




def main():
    args,_ = _parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    data_name, model_name, sampling, number, seed_list, loss_name = numbering(
        args.data_name, args.model, args.sampling, args.minor_number, args.seed, args.loss
    )

    # ✅ cross-domain 평가용 test 도메인 후보 리스트
    test_domain_list = [
        '6_cardio', '14_glass', '15_Hepatitis', '10_cover'
    ]

    print(f'device: {device}, dataset: {data_name}, model: {model_name}')

    AUC_list, PRAUC_list = [], []

    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dataset = dl.CustomDataset(data_name, args.file_path, number=number,
                                   sampling_method=sampling, seed=seed)
        max_features = dataset.train_data.shape[1]

        # ✅ 모델 학습
        if model_name == 'mlp':
            model = MLPClassifier(max_features, loss_name=loss_name, lr=args.lr, gamma=args.gamma,
                                  alpha=args.alpha, beta=args.beta, loss_type=args.cbloss, device=device)
            
            
            model.fit(dataset, epoch=args.epochs, batch_size=args.batch_size)
        elif model_name == 'lr':
            model = LogisticRegression(max_iter=200)
            model.fit(dataset.train_data, dataset.train_y)

        elif model_name == 'iforest':  # ✅ Isolation Forest 추가
            print("[INFO] Isolation Forest 모델 사용 중...")
            model = IsolationForest(
                n_estimators=200,
                contamination='auto',
                random_state=42,
                n_jobs=-1
            )
            model.fit(dataset.train_data)

        elif model_name == 'dbscan':  # ✅ DBSCAN 추가
            print("[INFO] DBSCAN 모델 사용 중...")
            model = DBSCAN(
                eps=0.5,        # 거리 임계값 (기본값: 0.5)
                min_samples=5,  # 최소 이웃 샘플 수
                n_jobs=-1
            )
            model.fit(dataset.train_data)
            trained_feature_dim = dataset.train_data.shape[1]
            print(f"[INFO] DBSCAN 학습 feature 수: {trained_feature_dim}")

        Ft.save_model(model, args.model_path, data_name, model_name, sampling, loss_name)

        # ✅ cross-domain 평가 loop
        for test_name in test_domain_list:
            if test_name == data_name:
                continue

            print(f"\n[TEST] Evaluating on {test_name} domain")

            test_dataset = dl.CustomDataset(test_name, args.file_path, number=number,
                                            sampling_method='none', seed=seed)

            # ✅ 평가
            y_pred = model_predict(model_name, model, test_dataset.test_data, trained_feature_dim=max_features)
            result = Ft.evaluation_metric(test_dataset.test_y, y_pred)

            print(f"Domain {test_name} → AUC: {result['auc']:.4f}, PRAUC: {result['prauc']:.4f}")
            AUC_list.append(result['auc'])
            PRAUC_list.append(result['prauc'])

    Ft.result_save(dataset, AUC_list, PRAUC_list, args.save_path,
                   {'data_name': data_name, 'model': model_name, 'sampling': sampling,
                    'number': number, 'loss': loss_name, "seed": seed_list,})
    print('Finish!')


if __name__ == '__main__':
    args,_ = _parse_args()
    main()
