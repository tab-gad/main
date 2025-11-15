import pandas as pd
import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from model.relational import make_correlation_features


#__________________________________________________________

#           Imbalanced Classification
#               0. satellite (31.64%)
#               1. cardio (9.61%)
#               2. Pageblocks (9.46%)
#               3. annthyroid (7.42%)
#               4. shuttle (7.15%)
#               5. thyroid (2.47%)
#               6. satimage-2 (1.22%)
#               7. Creditcard - binary class (0.17%)
#__________________________________________________________

def minor_class_number(data, label, number):

    # import pdb
    # pdb.set_trace()
    
    indices_x1 = np.where(label == 1)[0]
    indices_x0 = np.where(label == 0)[0]

    selected_x0 = data[indices_x0]

    selected_indices = np.random.choice(indices_x1, number, replace=False)
    selected_x1 = data[selected_indices]

    balanced_data = np.concatenate([selected_x0, selected_x1], axis=0)
    balanced_label = np.concatenate([label[indices_x0], label[selected_indices]], axis=0)

    if number == 1:
        balanced_data = np.concatenate([selected_x0, selected_x1, selected_x1], axis=0)
        balanced_label = np.concatenate([label[indices_x0], label[selected_indices], label[selected_indices]], axis=0)

    return balanced_data, balanced_label


class CustomDataset(Dataset):
    def __init__(self, dataset, file_path, number, sampling_method, seed, plot=False, max_features=30):
        real_data = np.load(file_path + dataset + '.npz')

        self.data = real_data['X']
        self.labels = real_data['y']
        self.seed = seed

        # 데이터 분할
        train_data, val_data, train_labels, val_labels = train_test_split(
            self.data, self.labels, test_size=0.2, stratify=self.labels, random_state=self.seed)
        val_data, test_data, val_labels, test_labels = train_test_split(
            val_data, val_labels, test_size=0.5, stratify=val_labels, random_state=self.seed)

        # minor class 조정
        if number == 'none':
            balanced_train_data = train_data
            balanced_train_labels = train_labels
        else:
            balanced_train_data, balanced_train_labels = minor_class_number(train_data, train_labels, number)

        # 정규화
        self.scaler = StandardScaler()
        balanced_train_data = self.scaler.fit_transform(balanced_train_data)
        val_data = self.scaler.transform(val_data)
        test_data = self.scaler.transform(test_data)

        # ✅ correlation 기반 feature 생성
        print(f"[INFO] Correlation feature 생성 시작 ({dataset}, max_features={max_features})")

        # Train
        df_train = pd.DataFrame(balanced_train_data, columns=[f"col{i}" for i in range(balanced_train_data.shape[1])])
        df_train_corr = make_correlation_features(df_train, max_features=max_features)
        balanced_train_data = np.array(df_train_corr)

        # Validation
        df_val = pd.DataFrame(val_data, columns=[f"col{i}" for i in range(val_data.shape[1])])
        df_val_corr = make_correlation_features(df_val, max_features=max_features)
        val_data = np.array(df_val_corr)

        # Test
        df_test = pd.DataFrame(test_data, columns=[f"col{i}" for i in range(test_data.shape[1])])
        df_test_corr = make_correlation_features(df_test, max_features=max_features)
        test_data = np.array(df_test_corr)

        print(f"[INFO] Correlation feature 생성 완료 → 최종 feature 수: {balanced_train_data.shape[1]}개\n")

        # PyTorch tensor 변환
        self.train_data = balanced_train_data
        self.val_data = val_data
        self.test_data = test_data

        self.train_y = torch.FloatTensor(balanced_train_labels)
        self.val_y = torch.FloatTensor(val_labels)
        self.test_y = torch.FloatTensor(test_labels)

        self.train_dataset = TensorDataset(torch.FloatTensor(balanced_train_data),
                                           torch.FloatTensor(balanced_train_labels))
        self.val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.FloatTensor(val_labels))
        self.test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
