import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Streamlit 앱 제목
st.title('대출 상환 여부 딥러닝 모델 🤑')

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    data = pd.read_csv('dataset/loan_data.csv')  # 데이터 파일 경로 확인 필요
    data = pd.get_dummies(data, columns=['purpose'], prefix='purpose')
    
    # 데이터 불균형 처리
    not_fully_paid_0 = data[data['not.fully.paid'] == 0]
    not_fully_paid_1 = data[data['not.fully.paid'] == 1]
    df_minority_upsampled = resample(not_fully_paid_1, replace=True, n_samples=8045)
    data = pd.concat([not_fully_paid_0, df_minority_upsampled])
    data = data.sample(frac=1).reset_index(drop=True)  # 데이터 셔플링
    return data

# 데이터 로드
data = load_data()

# 데이터의 특성 확인
st.write('전처리 완료된 데이터:')
st.write(data.head())

# 히트맵 시각화
st.subheader('특성 간 상관 관계 히트맵')
matrix = data.corr()
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# X, y 데이터 분할
X = data.drop("not.fully.paid", axis=1)
y = data["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=108, stratify=y)

# 데이터 정규화
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 모델 정의
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_s.shape[1],)),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Streamlit 위젯으로 배치 크기 및 에폭 수 입력 받기
batch_size = st.slider('Batch Size', min_value=16, max_value=128, step=16, value=64)
epochs = st.slider('Epochs', min_value=10, max_value=100, step=10, value=10)

# 모델 훈련
@st.cache_data
def train_model(batch_size, epochs):
    result = model.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=0
    )
    return result

# 훈련 버튼
train_button = st.button('모델 훈련')
if train_button:
    st.write("훈련 중...")
    history = train_model(batch_size, epochs)
    st.write("훈련 완료!")
    st.write(f"훈련 정확도: {history.history['accuracy'][-1]}")
    st.write(f"검증 정확도: {history.history['val_accuracy'][-1]}")

    # 훈련과 검증 정확도 시각화
    st.subheader('훈련 및 검증 정확도 그래프')
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

# 예측 및 결과 출력
predict_button = st.button('예측 실행')
if predict_button:
    predictions = (model.predict(X_test_s) > 0.5).astype("int32")  # 임계값 0.5 기준
  
    # ROC Curve 시각화
    y_pred_prob = model.predict(X_test_s)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    st.subheader('ROC Curve')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 대각선
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
