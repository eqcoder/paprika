import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
# 데이터 그룹화 및 평균 계산
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
data=pd.read_excel("생육통합데이터_라벨링.xlsx")

grouped_df = data.groupby(['농가레이블', '정식후일자'], as_index=False).agg({
    '일일생장률': 'mean',
    '줄기굵기': 'mean',
    '화방높이': 'mean',
    '엽장': 'mean',
    '마디별착과수': 'mean',
    '마디별열매수': 'mean',
    '생장강도': 'mean'
})

# 2. 생장상태 재계산
grouped_df['생장상태'] = grouped_df['생장강도'].apply(
    lambda x: '영양생장' if x < 0 else '생식생장'
)

# 3. 컬러맵 설정
cmap = LinearSegmentedColormap.from_list('생장강도', 
                                        ["#0D1752", "#12206E", "#FFFFFF", "#911616", "#810909"])

# 4. 전체 강도 범위 계산
min_intensity = grouped_df['생장강도'].min()
max_intensity = grouped_df['생장강도'].max()

# 5. 지표 리스트 정의
indicators = ['일일생장률', '줄기굵기', '화방높이', '엽장', '마디별착과수', '마디별열매수']

# 6. 농가별 시각화
for farm in grouped_df['농가레이블'].unique():
    farm_data = grouped_df[grouped_df['농가레이블'] == farm].copy()
    farm_data = farm_data.sort_values('정식후일자')
    
    # 새로운 창 생성 (지표별 서브플롯)
    fig, axs = plt.subplots(len(indicators), 1, figsize=(14, 12), sharex=True)
    plt.suptitle(f'농가 {farm} - 지표별 생장 추이', fontsize=16, y=0.99)
    
    # 각 지표별 서브플롯에 데이터 플로팅
    for i, indicator in enumerate(indicators):
        ax = axs[i]
        
        # 지표 데이터 플롯
        ax.plot(farm_data['정식후일자'], farm_data[indicator], 
                'o-', linewidth=2, markersize=8, color='#333333')
        ax.set_title(indicator, fontsize=14)
        ax.set_ylabel(indicator, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 생장 강도에 따른 배경색
        for j in range(len(farm_data)-1):
            start = farm_data['정식후일자'].iloc[j]
            end = farm_data['정식후일자'].iloc[j+1]
            intensity = farm_data['생장강도'].iloc[j]
            
            # 정규화된 강도 (0~1)
            norm_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)
            color = cmap(norm_intensity)
            ax.axvspan(start, end, color=color, alpha=0.3)
    
    # 공통 설정
    axs[-1].set_xlabel('정식 후 경과 일수', fontsize=12)
    
    # X축 범위 설정
    x_min = farm_data['정식후일자'].min() - 1
    x_max = farm_data['정식후일자'].max() + 1
    for ax in axs:
        ax.set_xlim(x_min, x_max)
    
    # 색상바 추가
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_intensity, vmax=max_intensity))
    cbar = fig.colorbar(sm, cax=cax, label='생장 강도')
    
    # 범례 텍스트 추가
    fig.text(0.93, 0.88, '생식생장', color='#911616', fontsize=12, ha='left', weight='bold')
    fig.text(0.93, 0.10, '영양생장', color='#0D1752', fontsize=12, ha='left', weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96], pad=2)
    plt.show()