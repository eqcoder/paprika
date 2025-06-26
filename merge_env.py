import pandas as pd
import glob
import os
from datetime import datetime, timedelta
import numpy as np

# 생육데이터 로드
growth_data = pd.read_excel('조사일_이전조사일_포함.xlsx')
growth_data['조사일자'] = pd.to_datetime(growth_data['조사일자']).dt.date

# 환경데이터 파일 목록 가져오기
env_files = glob.glob('data/기상환경/*.xlsx')
env_summary_list = []

for file in env_files:
    df_env = pd.read_excel(file)
    farm_name = df_env["농가명"].iloc[0]
    date_col = '측정시간'
    df_env['datetime'] = pd.to_datetime(df_env[date_col])
    df_env['date'] = df_env['datetime'].dt.date
    df_env['hour'] = df_env['datetime'].dt.hour
    
    # 필수 컬럼 확인
    required_cols = ['온도_외부']
    for col in required_cols:
        if col not in df_env.columns:
            raise ValueError(f'{file}에 {col} 컬럼이 없습니다.')
    
    # 1. 일별 요약 계산
    daily_data = df_env.groupby('date').agg(
        일평균온도=('온도_내부', 'mean'),
        일평균상대습도=('상대습도_내부', 'mean')
    ).reset_index()
    
    # 2. 주야간 온도차 계산
    주간 = df_env[(df_env['hour'] >= 6) & (df_env['hour'] <= 18)]
    야간 = df_env[(df_env['hour'] < 6) | (df_env['hour'] > 18)]
    
    주간온도 = 주간.groupby('date')['온도_내부'].mean().reset_index(name='주간평균온도')
    야간온도 = 야간.groupby('date')['온도_내부'].mean().reset_index(name='야간평균온도')
    
    온도차 = pd.merge(주간온도, 야간온도, on='date')
    온도차['주야간온도차'] = 온도차['주간평균온도'] - 온도차['야간평균온도']
    
    # 3. 일누적일사량 추출
    df_env_sorted = df_env.sort_values(['date', 'hour'], ascending=[True, False])
    last_entries = df_env_sorted.drop_duplicates('date', keep='first')
    누적일사량 = last_entries[['date', '누적일사량_외부']].rename(columns={'누적일사량_외부': '일누적일사량'})
    
    # 4. 데이터 통합
    daily_env = pd.merge(daily_data, 온도차[['date', '주야간온도차']], on='date')
    daily_env = pd.merge(daily_env, 누적일사량, on='date')
    daily_env['농가명'] = farm_name
    env_summary_list.append(daily_env)

# 환경데이터 통합
env_summary = pd.concat(env_summary_list, ignore_index=True)
env_summary.rename(columns={'date': '조사일자'}, inplace=True)

# 생육-환경 데이터 병합
final_data = pd.merge(
    growth_data,
    env_summary,
    on=['농가명', '조사일자'],
    how='left'
)

# ====== 결측치 처리 로직 추가 (이전/이후 일자 평균) ======
def fill_missing_with_neighbors(df, group_col, date_col, target_cols):
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)
    
    for col in target_cols:
        for farm in df[group_col].unique():
            farm_mask = df[group_col] == farm
            farm_data = df.loc[farm_mask].copy()
            
            # 결측치 인덱스 추출
            missing_idx = farm_data[farm_data[col].isnull()].index
            
            for idx in missing_idx:
                current_date = df.loc[idx, date_col]
                
                # 이전/이후 1일 범위 설정
                prev_date = current_date - timedelta(days=1)
                next_date = current_date + timedelta(days=1)
                
                # 이전/이후 일자 데이터 조회
                prev_vals = df[(df[group_col] == farm) & 
                               (df[date_col] == prev_date)][col]
                
                next_vals = df[(df[group_col] == farm) & 
                               (df[date_col] == next_date)][col]
                
                # 평균값 계산
                valid_vals = []
                if not prev_vals.empty: 
                    valid_vals.append(prev_vals.values[0])
                if not next_vals.empty: 
                    valid_vals.append(next_vals.values[0])
                
                if valid_vals:
                    fill_val = sum(valid_vals) / len(valid_vals)
                    df.loc[idx, col] = fill_val
                    
    return df

# 결측치 처리 적용
target_cols = ['일평균온도', '일평균상대습도', '주야간온도차', '일누적일사량']
final_data_filled = fill_missing_with_neighbors(
    final_data, 
    '농가명', 
    '조사일자', 
    target_cols
)

# ====== 결과 확인 및 저장 ======
print("결측치 처리 후 샘플 데이터:")
print(final_data_filled[['농가명', '조사일자'] + target_cols].head())

# 추가 보간 (필요시)
for col in target_cols:
    final_data_filled[col] = final_data_filled.groupby('농가명')[col].transform(
        lambda x: x.interpolate(method='linear')
    )

# 파일 저장
final_data_filled.to_excel('생육환경_통합데이터_결측치처리완료_최종.xlsx', index=False)
print("\n파일 저장 완료: 생육환경_통합데이터_결측치처리완료.xlsx")
