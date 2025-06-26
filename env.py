import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
data = pd.read_excel("생육환경_통합데이터_결측치처리완료_최종.xlsx")
columns = ['일평균온도', '일평균상대습도', '주야간온도차', '일누적일사량']
missing_counts = data[columns].isnull().sum()
print("환경데이터 결측치 개수:")
print(missing_counts)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 농가별 고유 색상 매핑 생성
unique_farms = data['농가레이블'].unique()
farm_palette = dict(zip(unique_farms, sns.color_palette('Set2', len(unique_farms))))

for i, col in enumerate(columns):
    overall_mean = data[col].mean()
    plt.figure(figsize=(12, 8))
    
    # 농가별로 색상 지정 (hue 사용)
    sns.boxplot(
        x='농가레이블',  # x축을 농가로 설정
        y=col,
        hue='농가레이블',  # 농가별로 다른 색상 적용
        data=data,
        palette=farm_palette,  # 미리 정의한 팔레트 사용
        dodge=False  # 동일 x축 위치에 박스플롯 표시
    )
    
    plt.axhline(
        y=overall_mean, 
        color='red', 
        linestyle='--', 
        linewidth=2,
        label=f'전체 평균: {overall_mean:.2f}'
    )
    plt.title(f'농가별 {col} 분포', fontsize=14)
    plt.xlabel('농가', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend([],[], frameon=False)
    plt.savefig(f'{col}_boxplot.png', dpi=300)  # 고해상도 저장
    plt.show()
