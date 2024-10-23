# healthcare-analysis-ibm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
#!/bin/bash
data = pd.read_csv('/content/heart.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Data Preprocessing
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Encode categorical variables if necessary (not required for this dataset)
# Normally you would use pd.get_dummies() or LabelEncoder

# Exploratory Data Analysis (EDA)
# Pairplot to visualize relationships
sns.pairplot(data, hue='target')
plt.title("Pairplot of Health Data")
plt.show()

# Feature selection
# Splitting the dataset into features and target
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plotting feature importance
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()
