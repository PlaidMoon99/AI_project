# pip install ydata-profiling
# pip install setuptools

import pandas as pd
from ydata_profiling import ProfileReport

# 데이터 불러오기
df = pd.read_csv('C:/AI_project/dataset/drug200.csv', encoding='cp949')

# 프로파일링 리포트 생성
profile = ProfileReport(
    df,
    title="EDA 보고서",
    explorative=True,
    html={
        'style': {
            'theme': 'united'  # 허용된 theme 중 하나로 변경
        }
    }
)

# 리포트 저장 및 출력
profile.to_file("C:/AI_project/dataset/report/drug200_eda_report.html")