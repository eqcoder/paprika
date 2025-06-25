import pandas as pd
import os
import glob
from datetime import datetime

# 1. 농가정보 파일 로드
try:
    farm_info = pd.read_excel('./data/농가정보.xlsx')
    print("농가정보 파일을 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("경고: '농가정보.xlsx' 파일을 찾을 수 없습니다. 빈 데이터프레임으로 대체합니다.")
    farm_info = pd.DataFrame()
print(farm_info.head())
# 2. 생육기본 폴더 내 모든 엑셀 파일 처리
growth_folder = './data/생육기본'
all_files = glob.glob(os.path.join(growth_folder, '*.xlsx'))
combined_data = pd.DataFrame()
i=0
for file_path in all_files:
    # 파일명에서 농가명 추출 (파일명 형식: [농가명]_생육기본.xlsx)
    i+=1 
    
    # 생육기본 데이터 로드
    growth_data = pd.read_excel(file_path)
    farm_name = growth_data.iloc[0]['농가명']
    # 조사년도 추출 (첫 행의 조사일자에서 연도 추출)
    if '조사일자' in growth_data.columns and not growth_data.empty:
        survey_year = pd.to_datetime(growth_data.iloc[0]['조사일자']).year
    else:
        survey_year = datetime.now().year
    
    # 농가정보에서 해당 농가 및 연도 일치하는 데이터 필터링
    if not farm_info.empty:
        farm_match = farm_info[
            (farm_info['농가명'] == farm_name) & 
            (pd.to_datetime(farm_info['구분날짜']).dt.year == survey_year)
        ]
        
        # 매칭된 정보 추출
        if not farm_match.empty:
            density = farm_match.iloc[0]['재식밀도']
            crop_type = farm_match.iloc[0]['품종']
            planting_date = farm_match.iloc[0]['정식기일자']
        else:
            density = crop_type = planting_date = '정보 없음'
    else:
        density = crop_type = planting_date = '정보 없음'
    
    # 새로운 열 추가
    growth_data['재식밀도'] = density
    growth_data['품종'] = crop_type
    growth_data['정식기일자'] = planting_date
    growth_data['농가명'] = farm_name
    growth_data['농가레이블']=i
    
    # 데이터 통합
    combined_data = pd.concat([combined_data, growth_data], ignore_index=True)

# 3. 통합 데이터 저장
if not combined_data.empty:
    output_file = './data/통합_생육기본_데이터.xlsx'
    combined_data.to_excel(output_file, index=False)
    print(f"총 {len(all_files)}개의 파일에서 {len(combined_data)}개 행을 처리하여 '{output_file}'에 저장했습니다.")
else:
    print("처리된 데이터가 없습니다. 생육기본 폴더를 확인해주세요.")

# 결과 미리보기
if not combined_data.empty:
    print("\n통합 데이터 미리보기:")
    print(combined_data[['농가명', '재식밀도', '품종', '정식기일자']].head())
