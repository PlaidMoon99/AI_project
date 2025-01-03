# library
import pandas as pd
import numpy as np  
import joblib           # 피클이라는 파일로 저장 ex)joblib.dump(model, 'diabetes_model.pkl')
import streamlit as st
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# font
plt.rcParams['font.family'] = 'NanumGothic'

# minus unicode
plt.rcParams['axes.unicode_minus'] = False

# 지수표현식
pd.options.display.float_format = '{:.2f}'.format

# data load
data = pd.read_csv('./dataset/diabetes.csv')

# select feature
selected_features = ['Glucose', 'BMI', 'Age']
X = data[selected_features]
y = data['Outcome'] # 예측할 대상

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# randomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# model save
joblib.dump(model, './py/diabetes_model.pkl')

# use test data for accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy : {accuracy*100:.2f}%')

# streamlit App
st.title('당뇨병 예측 시스템')
st.write('Glucose, BMI, Age 값을 입력하여 당뇨병 예측을 해보세요.')

# 사용자 입력받기
glucose = st.slider('Glucose (혈당 수치)', min_value=0, max_value=200, value=100)
bmi = st.slider('BMI (체질량지수)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
age = st.slider('Age (나이)', min_value=0, max_value=100, value=30)

# 예측하기 버튼
if st.button('예측하기'):
    # 입력값을 model에 전달
    model = joblib.load('./py/diabetes_model.pkl')
    input_data = np.array([[glucose, bmi, age]])
    prediction = model.predict(input_data)[0]
    
    # print result
    if prediction == 1:
        st.write('예측 결과: 당뇨병 가능성이 높습니다.')
    else:
        st.write('예측 결과: 당뇨병 가능성이 낮습니다.')
        