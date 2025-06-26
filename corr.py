import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df= pd.read_excel('생육환경_통합데이터_결측치처리완료_최종.xlsx')
df.drop(columns=['초장', '생장길이',  '농가명', '정식후일자', '이전_조사일', '개화마디', '착과마디', '열매마디', '수확마디', '농가레이블', '조사일자', '농가명', '개체번호', '줄기번호', '재식밀도', '품종', '정식기일자', '본주구분'], inplace=True, axis=1)
corr = df.corr()
def get_top_correlations(corr_matrix, n=10):
    """
    상위 n개 상관계수 쌍 추출 (절대값 기준)
    """
    # 대각선 및 중복 제거를 위한 마스크 생성
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_triu = corr_matrix.mask(mask)
    
    # 상관계수 페어 언스택
    corr_pairs = corr_triu.stack().reset_index()
    corr_pairs.columns = ['변수A', '변수B', '상관계수']
    
    # 절대값 기준 정렬
    corr_pairs['abs_상관계수'] = corr_pairs['상관계수'].abs()
    top_correlations = corr_pairs.sort_values('abs_상관계수', ascending=False).head(n)
    
    return top_correlations[['변수A', '변수B', '상관계수']]

# 상위 10개 상관계수 쌍 추출
top_corr_pairs = get_top_correlations(corr, n=10)

print("상위 10개 상관계수 쌍:")
print(top_corr_pairs.to_string(index=False))
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap='coolwarm',     # -1(음의 상관)은 파랑, 1(양의 상관)은 빨강
    annot=True,          # 각 셀에 상관계수 숫자 표시
    fmt=".2f",           # 소수점 둘째자리까지
    linewidths=0.5,      # 셀 경계선
    square=True          # 정사각형 셀
)
plt.title('열 간 상관계수 히트맵')
plt.tight_layout()
plt.show()

def get_top_correlations(corr_matrix, n=10):
    """
    상위 n개 상관계수 쌍 추출 (절대값 기준)
    """
    # 대각선 및 중복 제거를 위한 마스크 생성
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_triu = corr_matrix.mask(mask)
    
    # 상관계수 페어 언스택
    corr_pairs = corr_triu.stack().reset_index()
    corr_pairs.columns = ['변수A', '변수B', '상관계수']
    
    # 절대값 기준 정렬
    corr_pairs['abs_상관계수'] = corr_pairs['상관계수'].abs()
    top_correlations = corr_pairs.sort_values('abs_상관계수', ascending=False).head(n)
    
    return top_correlations[['변수A', '변수B', '상관계수']]

# 상위 10개 상관계수 쌍 추출
top_corr_pairs = get_top_correlations(corr, n=10)

print("상위 10개 상관계수 쌍:")
print(top_corr_pairs.to_string(index=False))