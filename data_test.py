import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, RobustScaler

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df= pd.read_excel('생육환경_통합데이터_결측치처리완료_최종.xlsx')
df.drop(columns=['초장', '생장길이',  '농가명', '일평균상대습도', '정식후일자', '이전_조사일', '개화마디', '착과마디', '열매마디', '수확마디', '농가레이블', '조사일자', '농가명', '개체번호', '줄기번호', '재식밀도', '품종', '정식기일자', '본주구분'], inplace=True, axis=1)
print(df.info())
# df['log_개화수'] = np.log1p(df['마디별 꽃수'])
# df['log_착과수'] = np.log1p(df['마디별착과수'])
# df['log_열매수'] = np.log1p(df['마디별열매수'])
# df['log_수확수'] = np.log1p(df['마디별수확수'])
# # Robust Scaling
# scaler = RobustScaler()
# columns= ['scaled_개화수', 'scaled_착과수', 'scaled_열매수', 'scaled_수확수']
# df[['scaled_개화수', 'scaled_착과수', 'scaled_열매수', 'scaled_수확수']] = scaler.fit_transform(df[['마디별 꽃수', '마디별착과수', '마디별열매수', '마디별수확수']])
skewness = df.skew()
print(f"왜도: {skewness}")
import seaborn as sns



scaler_std = StandardScaler()
morph_cols = ['엽수', '엽장', '엽폭', '마디별열매수', '주야간온도차', '줄기굵기', '일일생장률', '일평균온도', '일누적일사량']
df[morph_cols] = scaler_std.fit_transform(df[morph_cols])
df['화방높이'] = np.log1p(df['화방높이'])
scaler_robust = RobustScaler()
df['화방높이'] = scaler_robust.fit_transform(df[['화방높이']])
count_cols = ['마디별착과수', '마디별 꽃수', '마디별수확수']
for col in count_cols:
    df[col] = np.log1p(df[col])
    
scaler_std2 = StandardScaler()
df[['마디별착과수', '마디별 꽃수', '마디별수확수']] = scaler_std2.fit_transform(
    df[['마디별착과수', '마디별 꽃수', '마디별수확수']]
)
# def select_scaler(df, col, threshold=0.05):
#     # 이상치 비율 계산
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     outlier_ratio = ((df[col] < lower) | (df[col] > upper)).mean()
#     # 왜도 계산
#     skewness = df[col].skew()
    
#     # 스케일링 선택
#     if outlier_ratio > 0.03 and abs(skewness) > 1:
#         print(f"✅ {col}: 이상치 {outlier_ratio:.1%}, 왜도 {skewness:.2f} → log 변환 후 RobustScaler")
#     elif abs(skewness) > 0.5:
#         print(f"✅ {col}: 이상치 {outlier_ratio:.1%}, 왜도 {skewness:.2f} → log 변환 후 StandardScaler")
#     else:
#         print(f"⭕ {col}: 이상치 {outlier_ratio:.1%}, 왜도 {skewness:.2f} → StandardScaler")
    
# for col in df.columns:
#     select_scaler(df, col)
# 결과를 데이터프레임으로 변환
columns = df.columns
n_cols = len(columns)
total_plots = n_cols * 2  # 각 변수마다 2개 플롯 (히스토그램+KDE, Q-Q 플롯)
n_rows = (total_plots + 3) // 4  # 4열로 배치하기 위한 행 수 계산

fig, axes = plt.subplots(n_rows, 4, figsize=(20, n_rows*4))
axes = axes.flatten()  # 1차원 배열로 변환

plot_idx = 0
for i, col in enumerate(columns):
    # 왜도 계산
    skewness = df[col].skew()
    
    # 히스토그램 + KDE
    sns.histplot(df[col], kde=True, ax=axes[plot_idx], color='skyblue', bins=30)
    axes[plot_idx].set_title(f'{col} 히스토그램')
    axes[plot_idx].set_xlabel('값')
    axes[plot_idx].set_ylabel('빈도')
    plot_idx += 1
    
    # Q-Q 플롯
    sm.qqplot(df[col], line='45', ax=axes[plot_idx])
    axes[plot_idx].set_title(f'{col} - Q-Q 플롯')
    axes[plot_idx].set_xlabel('이론적 분위')
    axes[plot_idx].set_ylabel('표본 분위')
    plot_idx += 1

# 남는 서브플롯 숨기기
for j in range(plot_idx, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=6.0)
plt.show()

df.to_excel('생육환경_통합데이터_정규화.xlsx', index=False)