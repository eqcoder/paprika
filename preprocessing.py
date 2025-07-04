import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
# 품종별 생육 데이터 수집
def load_cultivar_data(cultivar_name):
    """품종별 데이터 로딩 함수"""
    # 실제 구현시 CSV/DB에서 품종별 데이터 로드
    data = pd.read_csv("통합_생육기본_데이터.xlsx")
    data['품종'] = cultivar_name
    return data
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
data = pd.read_excel("생육환경_통합데이터_결측치처리완료_최종.xlsx")
scaled_data=pd.read_excel('생육환경_통합데이터_정규화.xlsx')
numeric_cols = scaled_data.select_dtypes(include=['float64', 'int64']).columns
numeric_df = scaled_data[numeric_cols]


selected_features = ['엽수','화방높이','일일생장률','줄기굵기','엽장','엽폭']
X = scaled_data[selected_features]
pca = PCA(n_components=11)
pca_features = pca.fit_transform(scaled_data)
# 3. K-means 클러스터링 (2개 그룹)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # n_init 추가
clusters = kmeans.fit_predict(pca_features)

# 4. 클러스터 해석 (개선된 버전)
def interpret_clusters(df, clusters):
    """개선된 클러스터 해석 함수"""
    df_temp = scaled_data.copy()
    df_temp['cluster'] = clusters
    
    # 영양생장 지표: 엽수, 엽장, 엽폭, 줄기굵기
    veg_cols = ['엽수', '엽장', '엽폭', '줄기굵기']
    # 생식생장 지표: 마디별꽃수, 마디별착과수, 마디별열매수
    rep_cols = ['마디별 꽃수', '마디별착과수', '마디별열매수']
    
    # 클러스터별 평균 점수 계산
    cluster_stats = df_temp.groupby('cluster').mean()
    veg_scores = cluster_stats[veg_cols].mean(axis=1)
    rep_scores = cluster_stats[rep_cols].mean(axis=1)
    
    # 클러스터 매핑 생성
    cluster_mapping={}
    for cluster in cluster_stats.index:
        if veg_scores[cluster] > rep_scores[cluster]:
            cluster_mapping[cluster] = '영양생장'
        else:
            cluster_mapping[cluster] = '생식생장'
    
    df_temp['생장단계'] = df_temp['cluster'].map(cluster_mapping)
    return df_temp, cluster_mapping
result_df, cluster_mapping = interpret_clusters(scaled_data, clusters)
# 3. 생장 강도 계산 (클러스터 중심과의 거리 기반)
from sklearn.metrics.pairwise import euclidean_distances

# PCA 공간에서 클러스터 중심 좌표
centroids = kmeans.cluster_centers_

# 각 포인트의 클러스터 중심까지 거리 계산
distances = euclidean_distances(pca_features, centroids)

veg_cluster_id = [k for k, v in cluster_mapping.items() if v == '영양생장'][0]
rep_cluster_id = [k for k, v in cluster_mapping.items() if v == '생식생장'][0]

# 2. 각 포인트의 두 중심까지 거리 계산
d_veg = np.linalg.norm(pca_features - centroids[veg_cluster_id], axis=1)
d_rep = np.linalg.norm(pca_features - centroids[rep_cluster_id], axis=1)

# 3. 거리 차이 계산 (|d_veg - d_rep|)
dist_diff = np.abs(d_veg - d_rep)

# 4. 강도 계산 및 부호 할당
growth_intensity = np.where(d_veg < d_rep, 
                            -dist_diff,  # 영양생장이 더 가까울 때
                            dist_diff)   # 생식생장이 더 가까울 때
abs_max = np.max(np.abs(growth_intensity))

# 3. -1~1로 정규화
growth_intensity = growth_intensity / abs_max
# 5. 결과 할당
data['생장강도'] = growth_intensity
data['생장단계'] = np.where(d_veg < d_rep, '영양생장', '생식생장')


# 3. 생장 강도 계산 (클러스터 중심과의 거리 기반)
from sklearn.metrics.pairwise import euclidean_distances

# PCA 공간에서 클러스터 중심 좌표
centroids = kmeans.cluster_centers_

# 각 포인트의 클러스터 중심까지 거리 계산

data.to_excel("생육통합데이터_라벨링.xlsx")


# 6. 시각화 (PCA 공간에서)
color_dict = {'영양생장': '#1f77b4', '생식생장': '#d62728'}  # 파랑/빨강

plt.figure(figsize=(13, 10))

for label, color in color_dict.items():
    idx = data['생장단계'] == label
    plt.scatter(
        pca_features[idx, 0], pca_features[idx, 1],
        c=color, label=label, s=80, alpha=0.75, edgecolor='k'
    )

centers = kmeans.cluster_centers_[:, :2]
plt.scatter(
    centers[:, 0], centers[:, 1],
    c='gold', marker='X', s=300, edgecolor='black', linewidth=2, label='클러스터 중심'
)

# 변수 기여도(로딩) 화살표 (굵게, 폰트 크게)
loadings = pca.components_[:2].T
features = scaled_data.columns
for i, feature in enumerate(features):
    plt.arrow(
        0, 0, loadings[i, 0]*4, loadings[i, 1]*4,
        color='lightgreen', linewidth=3, head_width=0.18, head_length=0.25, alpha=0.7, length_includes_head=True
    )
    plt.text(
        loadings[i, 0]*4.5, loadings[i, 1]*4.5, feature,
        color='darkgreen', fontsize=12, fontweight='bold', ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3')
    )

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=16, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=16, fontweight='bold')
plt.title('PCA 기반 영양생장/생식생장 클러스터링', fontsize=20, fontweight='bold', pad=20)
plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.5)
plt.legend(title='생장 단계', fontsize=14, title_fontsize=15, loc='best', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout(pad=3.0)
plt.show()
# # 7. 주성분 분석 결과 출력
# print("="*50)
# print("주성분 분석 결과:")
# print(f"전체 설명 분산: {pca.explained_variance_ratio_.sum()*100:.2f}%")
# print(f"개별 설명 분산: {pca.explained_variance_ratio_}")

# 로딩(기여도) 분석
# loadings_df = pd.DataFrame(
#     pca.components_.T,
#     columns=[f'PC{i+1}' for i in range(11)],
#     index=X.columns
# )
# print("\n변수별 주성분 기여도:")
# print(loadings_df)
# # 클러스터 중심값 추출

# cluster_centers = pd.DataFrame(
#     kmeans.cluster_centers_,
#     columns=X.columns
# )
# thresholds = {}
# for col in cluster_centers.columns:
#     veg_value = cluster_centers.loc[0, col]
#     rep_value = cluster_centers.loc[1, col]
#     thresholds[col] = (veg_value + rep_value) / 2

# print("지표별 구분 임계값:")
# for k, v in thresholds.items():
#     scaled_value = [[v]]
#     if k=="엽수" or k=="엽장" or k=="엽폭" or k=="일일생장률" or k=="줄기굵기" or k=="화방높이":
#         scaler = StandardScaler()
#         scaler.fit(data[k].values.reshape(-1, 1))
        
#         restored_value = scaler.inverse_transform(scaled_value)
        
#     elif k=="화방높이":
#         X_log = np.log(data[k].values.reshape(-1, 1))

#         # 2단계: RobustScaler 적용
#         scaler = RobustScaler()
#         X_robust = scaler.fit_transform(X_log)

#         # --- 역변환 과정 ---
#         # 1. RobustScaler 역변환
#         restored_value = scaler.inverse_transform(scaled_value)

#         # 2. 로그 역변환
#         X_original_restored = np.exp(restored_value)
#     else:
#         X_log = np.log(data[k].values.reshape(-1, 1))

#         # 2단계: RobustScaler 적용
#         scaler = StandardScaler()
#         X_robust = scaler.fit_transform(X_log)

#         # --- 역변환 과정 ---
#         # 1. RobustScaler 역변환
#         restored_value = scaler.inverse_transform(scaled_value)

#         # 2. 로그 역변환
#         X_original_restored = np.exp(restored_value)
        
#     print(f"{k}: {restored_value}")
    
# # # 새로운 데이터 예측
# # # new_sample = {'cultivar': '페라리', 'stem_length': 75, 'leaf_count': 30, 
# # #               'flower_count': 4, 'fruit_count': 2}
# # # prediction = predict_growth_stage(new_sample, kmeans, cultivar_scalers, veg_cluster)
# # # print(f"예측 생장 단계: {prediction}")

