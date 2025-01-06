import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('dataset/drug200.csv')

# Label encoding
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['BP'] = encoder.fit_transform(df['BP'])
df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])
df['Drug'] = encoder.fit_transform(df['Drug'])

# Data preprocessing
X = df.drop("Drug", axis=1)
y = df["Drug"]

# Encoding categorical variables
X_transformed = pd.get_dummies(X, columns=["Sex", "BP", "Cholesterol"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# KNN Model
N_NEIGHBORS = 6
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X_transformed.columns,
    'Importance': knn.feature_importances_ if hasattr(knn, 'feature_importances_') else np.zeros(X_transformed.shape[1])
}).sort_values(by='Importance', ascending=False)

# Streamlit UI
st.title('💊Drug Classification with KNN')
st.write('This web app uses the K-Nearest Neighbors (KNN) model to predict drug classification based on various features.')

# Centralized Inputs and Button 사용자 입력
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Example of Sliders for input values
    age = st.slider('Age', min_value=10, max_value=100, value=50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    bp = st.selectbox('Blood Pressure', ['Normal', 'High', 'Low'])
    cholesterol = st.selectbox('Cholesterol', ['Normal', 'High'])

    # Encoding input values
    # sex_encoded = encoder.transform([sex])[0]
    # bp_encoded = encoder.transform([bp])[0]
    # cholesterol_encoded = encoder.transform([cholesterol])[0]

    # Prepare the input feature vector (one-hot encoding for categorical variables)
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex_Male': [1 if sex == 'Male' else 0],
        'Sex_Female': [1 if sex == 'Female' else 0],
        'BP_Normal': [1 if bp == 'Normal' else 0],
        'BP_High': [1 if bp == 'High' else 0],
        'BP_Low': [1 if bp == 'Low' else 0],
        'Cholesterol_Normal': [1 if cholesterol == 'Normal' else 0],
        'Cholesterol_High': [1 if cholesterol == 'High' else 0]
    })

    input_data_transformed = pd.get_dummies(input_data)
    
    # Align the columns with the training data columns
    input_data_transformed = input_data_transformed.reindex(columns=X_train.columns, fill_value=0)

    # 예측 버튼
    if st.button('예측하기'):
        # Prediction
        input_prediction = knn.predict(input_data_transformed)
     

        # Display Prediction
        st.subheader('Prediction Result')
        drug_mapping = {0: 'Drug 1', 1: 'Drug 2', 2: 'Drug 3', 3: 'Drug 4', 4: 'Drug 5', 5: 'Drug 6', 6: 'Drug 7', 7: 'Drug 8'}
        st.write(f"The predicted drug for the selected input is: {drug_mapping[input_prediction[0]]}")

# Display dataset
st.subheader('Dataset Preview')
st.write(df.head())

