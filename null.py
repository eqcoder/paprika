import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
df = pd.read_excel("./data/통합_생육기본_데이터.xlsx")
df['정식기일자'] = pd.to_datetime(df['정식기일자'])
df['조사일자'] = pd.to_datetime(df['조사일자'])
df['정식후일자'] = (df['조사일자'] - df['정식기일자']).dt.days
df.dropna(subset=["줄기굵기", "엽수", "엽장", "엽폭", "화방높이", "개화마디"], inplace=True)
df.drop(columns=["비고", "열1"], inplace=True)
df = df[df['화방높이'] != 0]

def get_date_map(group):
    group = group.sort_values().unique()
    return {group[i]: group[i - 1] if i > 0 else pd.NaT for i in range(len(group))}

# 농가별 이전 조사일 맵 딕셔너리 생성
farm_date_maps = {
    farm: get_date_map(group['조사일자'])
    for farm, group in df.groupby('농가레이블')
}

# 각 행에 해당하는 이전 조사일 찾아주는 함수
def lookup_prev_date(row):
    farm = row['농가레이블']
    date = row['조사일자']
    return farm_date_maps[farm].get(date, pd.NaT)

# 적용
df['이전_조사일'] = df.apply(lookup_prev_date, axis=1)
df['일일생장률'] = df["생장길이"]/(df['조사일자'] - df['이전_조사일']).dt.days
df.dropna(subset=["일일생장률"], inplace=True)
print(df.info())
print("결측치 개수 : ", df.isnull().sum().sum())

# 설정 재적용
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# 품종에 따른 색상 맵 만들기
품종색상 = dict(zip(df['품종'].unique(), sns.color_palette("tab10", n_colors=df['품종'].nunique())))

# 복사본 생성
df_outlier_flagged = df.copy()

# 분석할 지표 목록
columns = ['일일생장률', '화방높이', "줄기굵기", "엽수", "엽장", "엽폭", "개화마디", "착과마디", "열매마디", "수확마디", "마디별 꽃수", "마디별착과수", "마디별열매수", "마디별수확수"]

# 2행 8열 subplot 생성
fig, axes = plt.subplots(2, 7, figsize=(12, 8))
axes = axes.flatten()  # 1차원 배열로 변환

# boxplot 색상 지정
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C299FF', '#FF6666', '#66FFCC', '#CC99FF', '#FF9966', '#66CCFF', '#99FFCC', '#FFCC66', '#CC66FF', '#FF6699']

for i, col in enumerate(columns):
    df.boxplot(
        column=col,
        ax=axes[i],
        patch_artist=True,
        boxprops=dict(facecolor=colors[i])
    )
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# 남는 subplot 숨기기 (2행 8열이므로 16개, 컬럼은 14개라 2개 숨김)
for i in range(len(columns), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()
# 각 지표별 이상치 탐지
for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    # 이상치 여부 컬럼 생성
    df_outlier_flagged[f'{col}_이상치'] = (df[col] < lower_bound) | (df[col] > upper_bound)

# ✅ 하나라도 이상치인 행만 필터링
outlier_rows = df_outlier_flagged[
    df_outlier_flagged[[f'{col}_이상치' for col in columns]].any(axis=1)
]

# ✅ 이상치인 행 출력
print(f"총 이상치 포함 행 수: {len(outlier_rows)}")
print(outlier_rows.head())
df.to_excel("조사일_이전조사일_포함.xlsx", index=False)

# 시각화할 지표 목록
# 지표목록 = ['생장길이', '줄기굵기', '마디별 꽃수', '마디별착과수', '엽수']
# plt.figure(figsize=(14, 8))
# sns.boxplot(data=df, x='품종', y='생장길이',palette='Set2',  # 색상 팔레트
#     width=0.6,
#     fliersize=3,     # 이상치 점 크기
#     linewidth=1.5 )
# plt.title('품종별 생장길이 분포')
# plt.show()
# 각 지표별로 그래프 생성
# for 지표 in 지표목록:
#     plt.figure(figsize=(10, 6))
#     # 농가별로 라인 시각화
#     for 농가명, 농가데이터 in df.groupby('농가명'):
#         품종명 = 농가데이터['품종'].iloc[0]  # 농가별 품종 하나라고 가정
#         색상 = 품종색상[품종명]

#         sns.lineplot(
#             data=농가데이터.sort_values('정식후일자'),
#             x='정식후일자',
#             y=지표,
#             label=f"{농가명} ({품종명})",
#             color=색상
#         )

#     plt.title(f"정식 후 일자별 {지표} 변화")
#     plt.xlabel("정식 후 일수 (일)")
#     plt.ylabel(지표)
#     plt.legend(title="농가 (품종)", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()
