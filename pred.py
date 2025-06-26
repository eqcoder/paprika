import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# 데이터 준비 (가정: df는 정식후일자 기준 정렬된 데이터)
df = pd.read_excel("생육통합데이터_라벨링.xlsx")
grouped_df = df.groupby(['농가레이블', '정식후일자'], as_index=False).agg({
    '마디별 꽃수': 'mean',
    '마디별착과수': 'mean',
    '마디별수확수': 'mean',
    '마디별열매수': 'mean',
    '일평균온도': 'mean',
    '주야간온도차': 'mean',
    '일누적일사량': 'mean',
    '생장강도': 'mean',
})
def add_lag_features(df, lag=1):
    """이동평균 대신 바로 전 데이터만 사용하는 lag 특성 추가"""
    df = df.sort_values(['농가레이블', '정식후일자'])
    cols_to_lag = ['마디별 꽃수', '마디별착과수', '마디별수확수', '마디별열매수', '일평균온도', '주야간온도차', '일누적일사량']
    
    for col in cols_to_lag:
        df[f'{col}_lag{lag}'] = df.groupby('농가레이블')[col].shift(lag)
    
    return df.dropna()

# Lag 특성 추가
model_df = add_lag_features(grouped_df.copy(), lag=1)

# 특성과 타겟 분리
features = model_df.filter(like='_lag').columns.tolist()
X = model_df[features]
y = model_df['생장강도']
print(X.shape)
tscv = TimeSeriesSplit(n_splits=5)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# 교차 검증 수행
predictions = []
actuals = []
farm_ids = []
test_indices_list = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    predictions.extend(pred)
    test_indices_list.extend(test_index)
    actuals.extend(y_test.values)
    farm_ids.extend(model_df.iloc[test_index]['농가레이블'].values)

# 예측 결과 저장
result_df = pd.DataFrame({
    '농가레이블': farm_ids,
    '정식후일자': model_df.iloc[test_indices_list]['정식후일자'].values,
    '실제_생장강도': actuals,
    '예측_생장강도': predictions
})

def adjust_growth_state(row):
    """K-means 생장강도와 예측값 비교해 조정 방향 결정"""
    actual = row['실제_생장강도']
    pred = row['예측_생장강도']
    diff = pred - actual
    
    # 조정 방향 결정
    if abs(diff) < 0.1:
        return '유지'
    elif diff > 0:
        return '생식생장 강화' if actual > 0 else '영양생장 강화'
    else:
        return '생식생장 약화' if actual > 0 else '영양생장 약화'

# 생장 단계 조정 적용
result_df['조정_방향'] = result_df.apply(adjust_growth_state, axis=1)

def growth_management_strategy(adjustment):
    """조정 방향에 따른 구체적 관리 전략"""
    strategies = {
        '유지': "현재 관리 체계 유지",
        '영양생장 강화': """
        - 광량 증가 (최대 200μmol/m²/s)
        - 주야간 온도차 확대 (주간 25℃/야간 18℃)
        - 질소비료 비율 증가
        """,
        '영양생장 약화': """
        - 광량 감소 (150μmol/m²/s 이하)
        - 주간 온도 23℃로 낮춤
        - 질소비료 비율 감소
        """,
        '생식생장 강화': """
        - CO2 농도 1000ppm으로 증가
        - 인산비료 비율 증가
        - 과실 부하 조절 (적과)
        """,
        '생식생장 약화': """
        - CO2 농도 800ppm으로 감소
        - 인산비료 비율 감소
        - 야간 온도 20℃로 상승
        """
    }
    return strategies.get(adjustment, "전략 미정")

# 관리 전략 추가
result_df['관리_전략'] = result_df['조정_방향'].apply(growth_management_strategy)

import matplotlib.pyplot as plt

# 농가별 조정 방향 분포
plt.figure(figsize=(10, 6))
result_df.groupby(['농가레이블', '조정_방향']).size().unstack().plot(
    kind='bar', stacked=True, colormap='coolwarm'
)
plt.title('농가별 생장 조정 방향 분포')
plt.ylabel('빈도수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 생장 강도 비교 시각화
# for farm in result_df['농가레이블'].unique():
#     farm_data = result_df[result_df['농가레이블'] == farm]
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(farm_data['정식후일자'], farm_data['실제_생장강도'], 'o-', label='실제 강도')
#     plt.plot(farm_data['정식후일자'], farm_data['예측_생장강도'], 's-', label='예측 강도')
    
#     # 조정 방향 표시
#     for i, row in farm_data.iterrows():
#         plt.text(row['정식후일자'], row['실제_생장강도']+0.05, 
#                 row['조정_방향'], ha='center', fontsize=9)
    
#     plt.title(f'농가 {farm} - 생장 강도 비교')
#     plt.xlabel('정식 후 경과 일수')
#     plt.ylabel('생장 강도')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}
evaluate_regression(result_df["실제_생장강도"], result_df["예측_생장강도"])
result_df.to_excel("생육통합데이터_피드백.xlsx")