import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 및 필터링
df = pd.read_excel('조사일_이전조사일_포함.xlsx')
품종별_농가수 = df.groupby('품종')['농가레이블'].nunique()
품종_2개이상 = 품종별_농가수[품종별_농가수 >= 3].index.tolist()
df_filtered = df[df['품종'].isin(품종_2개이상)]

# 품종별 색상 매핑
unique_varieties = df_filtered['품종'].unique()
palette = sns.color_palette('Set2', len(unique_varieties))
품종색상 = dict(zip(unique_varieties, palette))

# 지표 목록
지표목록 = ['초장','생장길이', '화방높이', '줄기굵기', '마디별 꽃수', '마디별착과수', '엽수', '엽장', '엽폭']

# 서브플롯 설정 (3x2 그리드)
fig, axes = plt.subplots(3, 3, figsize=(18, 20), sharex=True)  # x축 공유
axes_flat = axes.flatten()  # 2D 배열을 1D로 변환

# 각 지표별 그래프 그리기

# 2. 각 지표별 그래프 그리기

# 각 지표별 그래프 그리기 (제목 추가, 각 서브플롯 범례 제거)
for i, 지표 in enumerate(지표목록):
    ax = axes_flat[i]
    for 농가명, 농가데이터 in df_filtered.groupby('농가레이블'):
        품종명 = 농가데이터['품종'].iloc[0]
        색상 = 품종색상[품종명]
        sns.lineplot(
            data=농가데이터.sort_values('정식후일자'),
            x='정식후일자',
            y=지표,
            color=색상,
            ax=ax,
            linewidth=1.5,
            alpha=0.7
        )
    ax.set_title(f"{지표} 변화", fontsize=14)
    ax.set_xlabel("정식 후 일수 (일)", fontsize=10)
    ax.set_ylabel(지표, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)
 

# 통합 범례 생성
handles, labels = [], []
for 품종명, 색상 in 품종색상.items():
    handles.append(plt.Line2D([0], [0], color=색상, lw=3))
    labels.append(품종명)

# 범례를 그래프 바로 아래에 배치 (좀 더 위쪽으로)
fig.legend(
    handles, labels,
    title='품종',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.01),  # 0.02로 상단에 가깝게(그래프 바로 아래)
    ncol=2,
    fontsize=12
)

# 전체 레이아웃 조정 (아래쪽 여백 확보)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.subplots_adjust(bottom=0.15)  # bottom을 0.03 이상으로 확보
plt.subplots_adjust(hspace=0.2, wspace=0.3)  # 서브플롯 간 간격 조정
plt.show()