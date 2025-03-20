#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

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

# Store rows with missing Genre values before dropping them
df_missing = df[df['Genre'].isnull()].copy()

# Drop rows where Genre is missing
df = df.dropna(subset=['Genre'])

# Encode Genre labels
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])  # Convert categorical to numeric

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
print(f'Number of PCA components: {pca.n_components_}')

# Train-test split using PCA-transformed data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train and evaluate Logistic Regression on PCA-transformed data
log_model_pca = LogisticRegression(max_iter=20000, C=0.5, solver='liblinear')
log_model_pca.fit(X_train_pca, y_train_pca)
y_pred_log_pca = log_model_pca.predict(X_test_pca)
print('Logistic Regression Accuracy (PCA Features):', accuracy_score(y_test_pca, y_pred_log_pca))
print(classification_report(y_test_pca, y_pred_log_pca))

# Train and evaluate Logistic Regression on original features
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

log_model_orig = LogisticRegression(max_iter=20000, C=0.5, solver='liblinear')
log_model_orig.fit(X_train_orig, y_train_orig)
y_pred_log_orig = log_model_orig.predict(X_test_orig)

print('Logistic Regression Accuracy (Original Features):', accuracy_score(y_test_orig, y_pred_log_orig))
print(classification_report(y_test_orig, y_pred_log_orig))

# Train and Evaluate Support Vector Machine (SVM) on PCA data
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_pca, y_train_pca)
y_pred_svm = svm_model.predict(X_test_pca)
print('SVM Accuracy (PCA Features):', accuracy_score(y_test_pca, y_pred_svm))
print(classification_report(y_test_pca, y_pred_svm))

# Train Random Forest on PCA-transformed data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_pca, y)

# Predict missing genres
if not df_missing.empty:
    X_missing_scaled = scaler.transform(df_missing.drop(columns=['Genre']))
    X_missing_pca = pca.transform(X_missing_scaled)
    df_missing['Genre'] = rf_model.predict(X_missing_pca)

    # Convert predicted values back to original genre labels
    df_missing['Genre'] = le.inverse_transform(df_missing['Genre'])
    print("Missing genres predicted successfully.")

    # Reintegrate predicted rows back into the main dataset
    df_complete = pd.concat([df, df_missing], ignore_index=True)
else:
    df_complete = df.copy()

# Save the updated dataset
df_complete.to_csv(r'C:\Users\mb\Desktop\data science\algebre  lineaire\music_dataset_updated.csv', index=False)
print("Updated dataset saved successfully.")


# In[ ]:




