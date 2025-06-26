import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_excel('조사일_이전조사일_포함.xlsx')
variables = ['일일생장률', '화방높이', '줄기굵기', '엽수', '엽장', '엽폭']


# 각 지표별 그래프 그리기
for var in variables:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='품종', 
        y=var, 
        data=df,
        palette='Set2',
    )
    overall_mean = df[var].mean()
    plt.axhline(
        y=overall_mean, 
        color='red', 
        linestyle='--', 
        linewidth=2,
        label=f'전체 평균: {overall_mean:.2f}'
    )
    plt.title(f'{var} 분포')
    plt.xlabel('품종')
    plt.ylabel(var)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{var}_boxplot.png')  # 각 그림을 파일로 저장
    plt.show()

for var in variables:
    # 품종별 데이터 그룹화
    groups = [group[var].dropna().values for name, group in df.groupby('품종')]
    
    # 일원분산분석 수행
    f_stat, p_value = f_oneway(*groups)
    
    print(f"\n### {var}의 분산분석 결과 ###")
    print(f"F-통계량: {f_stat:.4f}")
    print(f"p-값: {p_value:.4f}")
    
    # 유의성 판단
    alpha = 0.05
    if p_value < alpha:
        print(f"결과: 품종별 {var} 차이가 통계적으로 유의함 (p < {alpha})")
    else:
        print(f"결과: 품종별 {var} 차이가 통계적으로 유의하지 않음")
