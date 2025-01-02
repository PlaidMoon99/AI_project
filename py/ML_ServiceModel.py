#library
import pandas as pd
from ydata_profiling import ProfileReport

# data load
df = pd.read_csv('./dataset/global_tech_salary.txt', delimiter=',')

# EDA
# 프로파일링 리포트 생성
profile = ProfileReport(
    df,
    title="Global Tech Salary 보고서",
    explorative=True,
    html={
        'style': {
            'theme': 'united'  # 허용된 theme 중 하나로 변경
        }
    }
)

# 리포트 저장 및 출력
profile.to_file("C:/AI_project/dataset/report/global_tech_salary_report.html")