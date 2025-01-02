#!/usr/bin/env python
# coding: utf-8

# In[1]:


# library
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


# title
st.title('iris 꽃 분류기')


# In[3]:


# data load
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target], name='species')


# In[4]:


# sidebar
st.sidebar.header('사용자 입력 파라미터')


# In[5]:


# 슬라이더로 하이퍼파라미터 조정
n_estimators = st.sidebar.slider('Tree 개수 : ', 1, 100, 10)
max_depth = st.sidebar.slider('Max depth : ', 1, 10, 3)


# In[6]:


# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# model training
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)


# In[8]:


# Accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# In[9]:


# result
st.write(f'Model accuracy : {accuracy * 100:.2f}%')


# In[12]:


# 특징 중요도 시각화
st.subheader('특징 중요도')


# In[14]:


feature_importance = pd.DataFrame({
    '특징' : X.columns,
    '중요도' : clf.feature_importances_
}).sort_values('중요도', ascending=False)


# In[15]:


st.bar_chart(feature_importance.set_index('특징'))


# In[ ]:




