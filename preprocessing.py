import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 품종별 생육 데이터 수집
def load_cultivar_data(cultivar_name):
    """품종별 데이터 로딩 함수"""
    # 실제 구현시 CSV/DB에서 품종별 데이터 로드
    data = pd.read_csv("통합_생육기본_데이터.xlsx")
    data['품종'] = cultivar_name
    return data

# 품종별 데이터 수집 (예시: 3개 품종)
cultivars = ['페라리', '스페셜', '쥬빌리']
all_data = pd.concat([load_cultivar_data(cult) for cult in cultivars])
def cultivar_specific_normalization(data):
    """품종별 특성 기반 정규화"""
    normalized_data = data.copy()
    
    # 품종별 그룹화
    grouped = data.groupby('cultivar')
    
    # 품종별 Z-score 정규화[1]
    for cultivar, group in grouped:
        cult_indices = group.index
        scaler = StandardScaler()
        normalized_data.loc[cult_indices, ['stem_length', 'leaf_count']] = scaler.fit_transform(
            normalized_data.loc[cult_indices, ['stem_length', 'leaf_count']])
    
    # 개화/과실 특성은 품종별 상대적 비율로 변환[4]
    for cultivar, group in grouped:
        cult_indices = group.index
        max_flower = group['flower_count'].max()
        max_fruit = group['fruit_count'].max()
        
        if max_flower > 0:
            normalized_data.loc[cult_indices, 'flower_ratio'] = (
                normalized_data.loc[cult_indices, 'flower_count'] / max_flower)
        
        if max_fruit > 0:
            normalized_data.loc[cult_indices, 'fruit_ratio'] = (
                normalized_data.loc[cult_indices, 'fruit_count'] / max_fruit)
    
    return normalized_data.fillna(0)

# 정규화 적용
normalized_data = cultivar_specific_normalization(all_data)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 클러스터링에 사용할 특징 선택
features = normalized_data[['줄기굵기', '엽수', '개화', '착과']]

# 차원 축소[1]
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features)

# K-means 클러스터링[2]
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(pca_features)

# 결과 통합
normalized_data['cluster'] = clusters
def interpret_clusters(normalized_data):
    """클러스터 결과 해석"""
    cluster_stats = normalized_data.groupby('cluster').mean()
    
    # 영양생장 클러스터 판별 기준
    if cluster_stats.loc[0, 'stem_length'] > cluster_stats.loc[1, 'stem_length']:
        veg_cluster = 0
        rep_cluster = 1
    else:
        veg_cluster = 1
        rep_cluster = 0
    
    # 라벨 매핑 생성
    normalized_data['growth_stage'] = normalized_data['cluster'].apply(
        lambda x: '영양생장' if x == veg_cluster else '생식생장')
    
    return normalized_data, veg_cluster, rep_cluster

# 클러스터 해석
result_data, veg_cluster, rep_cluster = interpret_clusters(normalized_data)
import matplotlib.pyplot as plt

# 품종별 클러스터 분포 시각화
plt.figure(figsize=(10, 6))
for cultivar in cultivars:
    cult_data = result_data[result_data['cultivar'] == cultivar]
    plt.scatter(
        cult_data['stem_length'], 
        cult_data['leaf_count'],
        label=f'{cultivar}',
        s=50, alpha=0.7
    )

# 클러스터 중심 표시
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Cluster Centers')

plt.xlabel('정규화 초장')
plt.ylabel('정규화 엽수')
plt.title('품종별 생장 단계 분류 결과')
plt.legend()
plt.grid(True)
plt.show()

# 품종별 분류 결과 출력
print("품종별 분류 비율:")
print(result_data.groupby(['cultivar', 'growth_stage']).size().unstack())
def predict_growth_stage(new_data, kmeans_model, scalers, veg_cluster):
    """
    새로운 데이터 분류 함수
    - new_data: {'cultivar': '페라리', 'stem_length': 85, ...}
    - scalers: 품종별 스케일러 딕셔너리
    """
    # 품종별 스케일러 선택
    scaler = scalers[new_data['cultivar']]
    
    # 수치형 특징 정규화
    num_features = ['stem_length', 'leaf_count']
    scaled_num = scaler.transform([[new_data[f] for f in num_features]])
    
    # 범주형 특징 처리
    max_flower = cultivar_max[new_data['cultivar']]['flower_count']
    max_fruit = cultivar_max[new_data['cultivar']]['fruit_count']
    
    flower_ratio = new_data['flower_count'] / max_flower if max_flower > 0 else 0
    fruit_ratio = new_data['fruit_count'] / max_fruit if max_fruit > 0 else 0
    
    # 특징 벡터 생성
    features = np.array([scaled_num[0,0], scaled_num[0,1], flower_ratio, fruit_ratio]).reshape(1, -1)
    
    # 클러스터 예측
    cluster = kmeans_model.predict(features)[0]
    
    return '영양생장' if cluster == veg_cluster else '생식생장'

# 스케일러 및 최대값 저장 (실제 구현시 학습 데이터 기반 계산)
cultivar_scalers = {cult: StandardScaler().fit(all_data[all_data['cultivar']==cult][['stem_length', 'leaf_count']]) 
                    for cult in cultivars}
cultivar_max = {cult: all_data[all_data['cultivar']==cult][['flower_count', 'fruit_count']].max().to_dict() 
                for cult in cultivars}

# 새로운 데이터 예측
new_sample = {'cultivar': '페라리', 'stem_length': 75, 'leaf_count': 30, 
              'flower_count': 4, 'fruit_count': 2}
prediction = predict_growth_stage(new_sample, kmeans, cultivar_scalers, veg_cluster)
print(f"예측 생장 단계: {prediction}")

