import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
scaled_data=pd.read_excel('생육환경_통합데이터_정규화.xlsx')
numeric_cols = scaled_data.select_dtypes(include=['float64', 'int64']).columns
numeric_df = scaled_data[numeric_cols]
# def cultivar_specific_normalization(data):
#     """품종별 특성 기반 정규화"""
#     normalized_data = data.copy()
    
#     # 품종별 그룹화
#     grouped = data.groupby('cultivar')
    
#     # 품종별 Z-score 정규화[1]
#     for cultivar, group in grouped:
#         cult_indices = group.index
#         scaler = StandardScaler()
#         normalized_data.loc[cult_indices, ['stem_length', 'leaf_count']] = scaler.fit_transform(
#             normalized_data.loc[cult_indices, ['stem_length', 'leaf_count']])
    
#     # 개화/과실 특성은 품종별 상대적 비율로 변환[4]
#     for cultivar, group in grouped:
#         cult_indices = group.index
#         max_flower = group['flower_count'].max()
#         max_fruit = group['fruit_count'].max()
        
#         if max_flower > 0:
#             normalized_data.loc[cult_indices, 'flower_ratio'] = (
#                 normalized_data.loc[cult_indices, 'flower_count'] / max_flower)
        
#         if max_fruit > 0:
#             normalized_data.loc[cult_indices, 'fruit_ratio'] = (
#                 normalized_data.loc[cult_indices, 'fruit_count'] / max_fruit)
    
#     return normalized_data.fillna(0)








# 클러스터링에 사용할 특징 선택
# PCA 적용 후 설명 분산 확인
# pca = PCA()
# pca.fit(numeric_df)  # 스케일링된 데이터

# 누적 설명 분산 계산
# explained_variance = pca.explained_variance_ratio_
# cumulative_variance = np.cumsum(explained_variance)

# # 시각화
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'o-')
# plt.axhline(y=0.95, color='r', linestyle='--')  # 95% 기준선
# plt.title('누적 설명 분산')
# plt.xlabel('주성분 개수')
# plt.ylabel('누적 설명 분산 비율')
# plt.grid(True)
# plt.show()

# # 최적의 주성분 개수 선택 (일반적으로 95% 이상)
# n_components = np.argmax(cumulative_variance >= 0.95) + 1
# print(f"95% 분산 설명을 위한 주성분 개수: {n_components}")
# explained_variance = pca.explained_variance_ratio_
# print(f"설명 분산: {explained_variance}")
# loadings = pd.DataFrame(
#     pca.components_.T,
#     columns=[f'PC{i+1}' for i in range(pca.n_components_)],
#     index=df.columns
# )
# print(loadings)
# # PC1에서 |로딩| > 0.5인 변수 추출
# pc1_important = loadings[abs(loadings['PC1']) > 0.5].index.tolist()
# print(f"PC1 주요 변수: {pc1_important}")
# plt.plot(explained_variance, 'o-')
# plt.axhline(y=0.05, color='r', linestyle='--') # 유의미한 분산 임계값
# plt.title("스크리 플롯: 분산 설명 비율")
# plt.xlabel("주성분")
# plt.ylabel("설명 분산")
# plt.show()
# plt.figure(figsize=(10, 6))
# sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
# plt.title("변수별 주성분 로딩")
# plt.show()
# 차원 축소[1]
# 2. PCA (11개 주성분)

selected_features = ['엽수','화방높이','일일생장률','줄기굵기','엽장','엽폭']
X = scaled_data[selected_features]
pca = PCA(n_components=6)
pca_features = pca.fit_transform(X)
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
    cluster_mapping = {}
    for cluster in cluster_stats.index:
        if veg_scores[cluster] > rep_scores[cluster]:
            cluster_mapping[cluster] = '영양생장'
        else:
            cluster_mapping[cluster] = '생식생장'
    
    df_temp['생장단계'] = df_temp['cluster'].map(cluster_mapping)
    return df_temp, cluster_mapping

# 5. 결과 통합 및 해석
# result_df, cluster_mapping = interpret_clusters(scaled_data, clusters)

# # 6. 시각화 (PCA 공간에서)
# color_dict = {'영양생장': '#1f77b4', '생식생장': '#d62728'}  # 파랑/빨강

# plt.figure(figsize=(13, 10))

# for label, color in color_dict.items():
#     idx = result_df['생장단계'] == label
#     plt.scatter(
#         pca_features[idx, 0], pca_features[idx, 1],
#         c=color, label=label, s=80, alpha=0.75, edgecolor='k'
#     )

# centers = kmeans.cluster_centers_[:, :2]
# plt.scatter(
#     centers[:, 0], centers[:, 1],
#     c='gold', marker='X', s=300, edgecolor='black', linewidth=2, label='클러스터 중심'
# )

# # 변수 기여도(로딩) 화살표 (굵게, 폰트 크게)
# loadings = pca.components_[:2].T
# features = scaled_data.columns
# for i, feature in enumerate(features):
#     plt.arrow(
#         0, 0, loadings[i, 0]*4, loadings[i, 1]*4,
#         color='lightgreen', linewidth=3, head_width=0.18, head_length=0.25, alpha=0.7, length_includes_head=True
#     )
#     plt.text(
#         loadings[i, 0]*4.5, loadings[i, 1]*4.5, feature,
#         color='darkgreen', fontsize=12, fontweight='bold', ha='center', va='center',
#         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3')
#     )

# plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=16, fontweight='bold')
# plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=16, fontweight='bold')
# plt.title('PCA 기반 영양생장/생식생장 클러스터링', fontsize=20, fontweight='bold', pad=20)
# plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.5)
# plt.legend(title='생장 단계', fontsize=14, title_fontsize=15, loc='best', frameon=True, facecolor='white', edgecolor='black')
# plt.tight_layout(pad=3.0)
# plt.show()
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

cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X.columns
)
thresholds = {}
for col in cluster_centers.columns:
    veg_value = cluster_centers.loc[0, col]
    rep_value = cluster_centers.loc[1, col]
    thresholds[col] = (veg_value + rep_value) / 2

print("지표별 구분 임계값:")
for k, v in thresholds.items():
    print(f"{k}: {v}")


# # # 새로운 데이터 예측
# # # new_sample = {'cultivar': '페라리', 'stem_length': 75, 'leaf_count': 30, 
# # #               'flower_count': 4, 'fruit_count': 2}
# # # prediction = predict_growth_stage(new_sample, kmeans, cultivar_scalers, veg_cluster)
# # # print(f"예측 생장 단계: {prediction}")

