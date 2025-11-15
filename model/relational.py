import itertools
import numpy as np
import pandas as pd
import math

def make_correlation_features(df: pd.DataFrame):
    """
    모든 가능한 2개/3개 변수 조합의 correlation 항을 계산해 feature로 확장.
    """
    cols = df.columns
    n_cols = len(cols)
    corr_features = {}

    print(f"[INFO] 현재 데이터 컬럼 수: {n_cols}개")

    # 2개~3개 조합 각각 correlation feature 생성
    for i in range(2, 4):
        print(f"[INFO] -> {i}-way correlation feature 개수: {math.comb(n_cols, i)}개")

        if i == 2:
            for (c1, c2) in itertools.combinations(cols, 2):
                corr_val = np.corrcoef(df[c1], df[c2])[0, 1]
                corr_features[f"corr_{c1}_{c2}"] = corr_val

        elif i == 3:
            for (c1, c2, c3) in itertools.combinations(cols, 3):
                # 세 변수 간 평균 correlation (단순화된 nC3 표현)
                m = df[[c1, c2, c3]].corr().values
                tri_corr = np.mean(m[np.triu_indices(3, 1)])  # 상삼각 평균
                corr_features[f"corr_{c1}_{c2}_{c3}"] = tri_corr

    # correlation 결과를 feature로 확장 (모든 샘플에 동일 값으로 broadcast)
    corr_df = pd.DataFrame(np.tile(list(corr_features.values()), (len(df), 1)),
                           columns=list(corr_features.keys()))
    
    # result = pd.concat([df.reset_index(drop=True), corr_df], axis=1)
    print(f"[INFO] 최종 feature 수: {corr_df.shape[1]}개")

    return corr_df

import itertools
import numpy as np
import pandas as pd
import math
import random

def make_correlation_feature_older(df: pd.DataFrame, max_features: int = 30):
    """
    모든 가능한 2개/3개 변수 조합의 correlation 항을 계산해 feature로 확장하되,
    전체 feature 수가 max_features(기본 30)를 넘지 않도록 제한.
    """
    cols = df.columns
    n_cols = len(cols)
    corr_features = {}

    print(f"[INFO] 현재 데이터 컬럼 수: {n_cols}개 (최대 {max_features}까지 확장 예정)")

    # 2개~3개 조합 각각 correlation feature 생성
    for i in range(2, 4):
        possible = math.comb(n_cols, i)
        print(f"[INFO] -> {i}-way correlation feature 개수: {possible}개")

        # i-way correlation 계산
        if i == 2:
            for (c1, c2) in itertools.combinations(cols, 2):
                corr_val = np.corrcoef(df[c1], df[c2])[0, 1]
                corr_features[f"corr_{c1}_{c2}"] = corr_val

        elif i == 3:
            for (c1, c2, c3) in itertools.combinations(cols, 3):
                m = df[[c1, c2, c3]].corr().values
                tri_corr = np.mean(m[np.triu_indices(3, 1)])
                corr_features[f"corr_{c1}_{c2}_{c3}"] = tri_corr

        # ✅ 현재까지 feature 수가 30 이상이면 stop
        if len(corr_features) >= max_features:
            print(f"[STOP] {i}-way 단계에서 feature 수 {len(corr_features)} ≥ {max_features}")
            break

    # ✅ feature cap 적용
    all_corr_names = list(corr_features.keys())
    all_corr_values = list(corr_features.values())

    # 만약 전체가 30개 이상이면 랜덤 샘플링
    if len(all_corr_names) > max_features:
        n_select = max_features
        select_idx = random.sample(range(len(all_corr_names)), n_select)
        selected_corr_names = [all_corr_names[j] for j in select_idx]
        selected_corr_values = [all_corr_values[j] for j in select_idx]
    else:
        selected_corr_names = all_corr_names
        selected_corr_values = all_corr_values

    # ✅ 선택된 correlation feature로 DataFrame 생성
    corr_df = pd.DataFrame(np.tile(selected_corr_values, (len(df), 1)),
                           columns=selected_corr_names)

    # result = pd.concat([df.reset_index(drop=True), corr_df], axis=1)

    print(f"[INFO] 최종 feature 수: {corr_df.shape[1]}개")
    return corr_df

import itertools
import numpy as np
import pandas as pd
import math
import random
def make_correlation_features(df: pd.DataFrame, max_features: int = 30):
    """
    모든 가능한 2개/3개 변수 조합의 correlation 항을 계산해 feature로 확장하되,
    NaN이나 inf 발생 시 0으로 대체하고,
    최종 feature는 correlation feature만 포함하며 총 max_features개로 제한.
    """
    cols = df.columns
    n_cols = len(cols)
    corr_features = {}

    print(f"[INFO] 현재 데이터 컬럼 수: {n_cols}개 (correlation feature만 {max_features}개까지 생성)")

    for i in range(2, 4):
        possible = math.comb(n_cols, i)
        print(f"[INFO] -> {i}-way correlation feature 개수: {possible}개")

        if i == 2:
            for (c1, c2) in itertools.combinations(cols, 2):
                corr_val = np.corrcoef(df[c1], df[c2])[0, 1]
                if np.isnan(corr_val) or np.isinf(corr_val):
                    corr_val = 0.0
                corr_features[f"corr_{c1}_{c2}"] = corr_val

        elif i == 3:
            for (c1, c2, c3) in itertools.combinations(cols, 3):
                m = df[[c1, c2, c3]].corr().values
                tri_corr = np.mean(m[np.triu_indices(3, 1)])
                if np.isnan(tri_corr) or np.isinf(tri_corr):
                    tri_corr = 0.0
                corr_features[f"corr_{c1}_{c2}_{c3}"] = tri_corr

        # 30개 이상이면 멈추기
        if len(corr_features) >= max_features:
            print(f"[STOP] {i}-way 단계에서 correlation feature {len(corr_features)}개 생성 (max {max_features})")
            break

    # ✅ 랜덤으로 30개 선택
    all_corr_names = list(corr_features.keys())
    all_corr_values = list(corr_features.values())

    if len(all_corr_names) > max_features:
        select_idx = random.sample(range(len(all_corr_names)), max_features)
        selected_corr_names = [all_corr_names[j] for j in select_idx]
        selected_corr_values = [all_corr_values[j] for j in select_idx]
    else:
        selected_corr_names = all_corr_names
        selected_corr_values = all_corr_values

    # ✅ correlation feature만 포함 (원래 column 제외)
    corr_df = pd.DataFrame(
        np.tile(selected_corr_values, (len(df), 1)),
        columns=selected_corr_names
    )

    corr_df = corr_df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"[INFO] 최종 correlation feature 수: {corr_df.shape[1]}개 (원래 column 제외)")
    return corr_df


import itertools
import numpy as np
import pandas as pd
import math
import random

def make_correlation_features(df: pd.DataFrame, max_features: int = 30):
    """
    1–10번째: 2개 변수 correlation
    11–20번째: 3개 변수 correlation
    21–30번째: 4개 변수 correlation
    NaN/inf는 0으로 대체하며, 최종적으로 correlation feature 30개만 생성.
    """
    cols = df.columns
    n_cols = len(cols)
    corr_features = {}

    print(f"[INFO] 현재 데이터 컬럼 수: {n_cols}개 (2,3,4-way 랜덤 correlation으로 {max_features}개 생성)")

    # ✅ 2-way correlation (1–10)
    comb2 = list(itertools.combinations(cols, 2))
    select2 = random.sample(comb2, min(10, len(comb2)))
    for (c1, c2) in select2:
        val = np.corrcoef(df[c1], df[c2])[0, 1]
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        corr_features[f"corr2_{c1}_{c2}"] = val

    # ✅ 3-way correlation (11–20)
    comb3 = list(itertools.combinations(cols, 3))
    select3 = random.sample(comb3, min(10, len(comb3)))
    for (c1, c2, c3) in select3:
        m = df[[c1, c2, c3]].corr().values
        tri_corr = np.mean(m[np.triu_indices(3, 1)])
        if np.isnan(tri_corr) or np.isinf(tri_corr):
            tri_corr = 0.0
        corr_features[f"corr3_{c1}_{c2}_{c3}"] = tri_corr

    # ✅ 4-way correlation (21–30)
    if n_cols >= 4:
        comb4 = list(itertools.combinations(cols, 4))
        select4 = random.sample(comb4, min(10, len(comb4)))
        for (c1, c2, c3, c4) in select4:
            m = df[[c1, c2, c3, c4]].corr().values
            quad_corr = np.mean(m[np.triu_indices(4, 1)])  # 상삼각 평균
            if np.isnan(quad_corr) or np.isinf(quad_corr):
                quad_corr = 0.0
            corr_features[f"corr4_{c1}_{c2}_{c3}_{c4}"] = quad_corr

    # ✅ correlation DataFrame 생성 (원래 column 제외)
    corr_df = pd.DataFrame(
        np.tile(list(corr_features.values()), (len(df), 1)),
        columns=list(corr_features.keys())
    )

    corr_df = corr_df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"[INFO] 최종 correlation feature 수: {corr_df.shape[1]}개 (2way:{len(select2)}, 3way:{len(select3)}, 4way:{min(10, len(comb4))})")

    return corr_df
