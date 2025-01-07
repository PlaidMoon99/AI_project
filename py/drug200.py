# library
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

#load dataset
df = pd.read_csv('dataset/drug200.csv')

df.head()

print("null 값 :\n", df.isnull().sum())
print("중복값 : ", df.duplicated().sum())

df.info()

# labelencoding
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['BP'] = encoder.fit_transform(df['BP'])
df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])
df['Drug'] = encoder.fit_transform(df['Drug'])

sns.heatmap(df.corr(), cmap='coolwarm', annot=True)

# randomForest
# X, y
X = df.drop('Drug', axis=1)
y = df['Drug']

# data split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.show()

# KNN
# preprocessing
X = df.drop("Drug", axis=1)
y = df["Drug"]
X_transformed = pd.get_dummies(X, columns=["Sex", "BP", "Cholesterol"])

print(X_transformed.head())

# data split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# model - KNN
N_NEIGHBORS = 6
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
knn.fit(X_train, y_train)

# prediction
y_pred = knn.predict(X_test)

# model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy}")

print(f"Confusion Matrix :\n {confusion_matrix(y_test, y_pred)}")

report = classification_report(y_test, y_pred)
print(f"Classification report :\n{report}")

# 예측값과 실제값 비교 그래프 
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Values', marker='o')
plt.plot(y_pred, label='Predicted Values', marker='x')
plt.legend()
plt.title('Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Target')
plt.show()