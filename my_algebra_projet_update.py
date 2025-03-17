#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import scipy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load dataset
df = pd.read_csv(r'C:\Users\mb\Desktop\data science\algebre  lineaire\music_dataset_mod.csv')

# Display basic info and first rows
print(df.info())
print(df.head())

# Load the data legend for feature understanding
legend = pd.read_excel(r'C:\Users\mb\Desktop\data science\algebre  lineaire\Music Data Legend.xlsx')
print(legend.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing Genre values
df = df.dropna(subset=['Genre'])

# Encode Genre labels
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])

# Define features and target
X = df.drop(columns=['Genre'])
y = df['Genre']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Correlation Matrix Visualization
plt.figure(figsize=(12,8))
sns.heatmap(pd.DataFrame(X_scaled, columns=X.columns).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Apply PCA
pca = PCA(n_components=0.85)  # Retain 85% variance
X_pca = pca.fit_transform(X_scaled)
print(f'Number of components: {pca.n_components_}')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train and Evaluate Logistic Regression
log_model = LogisticRegression(max_iter=20000, C=0.5, solver='liblinear')
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Train and Evaluate Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Predict missing genres
df_missing = df[df['Genre'].isnull()]
if not df_missing.empty:
    df_missing_scaled = scaler.transform(df_missing.drop(columns=['Genre']))
    df_missing_pca = pca.transform(df_missing_scaled)
    df_missing['Genre'] = rf_model.predict(df_missing_pca)  # Using Random Forest for better accuracy
    print("Missing genres predicted using Random Forest.")


# In[ ]:




