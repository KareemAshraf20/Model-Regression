# California Housing Price Prediction - Machine Learning Project

## 📖 Project Overview
This project implements various regression models to predict housing prices in California using the famous California Housing dataset. The goal is to compare different machine learning algorithms and preprocessing techniques to find the most effective approach for predicting median house values.

## 🏠 Dataset
The dataset contains information from the 1990 California census with the following features:
- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average number of household members
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude
- **MedHouseVal**: Median house value for California districts (target variable)

## 🛠️ Technologies Used
- Python 3
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 📊 Model Performance Comparison

| Model | R² Score | Notes |
|-------|----------|-------|
| Random Forest Regressor | 0.806 | Best performing model |
| Decision Tree Regressor | 0.616 | Good performance but prone to overfitting |
| Ridge Regression (α=100) | 0.581 | Regularized linear model |
| Lasso Regression (α=1) | 0.284 | Sparse model with feature selection |
| Polynomial Regression + PCA | 0.341 | Non-linear approach with dimensionality reduction |
| Support Vector Regression | 0.069 | Poor performance on this dataset |

## 📁 Code Structure

### 1. Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
```
*Import all necessary libraries for data manipulation, visualization, and machine learning modeling.*

### 2. Load and Explore Data
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
housing.frame
```
*Load the California Housing dataset as a pandas DataFrame for easy manipulation and exploration.*

### 3. Data Preparation
```python
x = housing.frame.drop('MedHouseVal', axis=1)
y = housing.frame['MedHouseVal']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
*Separate features (x) and target variable (y), then split into training and testing sets (80/20 split).*

### 4. Dimensionality Reduction with PCA
```python
pca = PCA(n_components=5)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
```
*Apply Principal Component Analysis to reduce dimensionality from 8 features to 5 principal components.*

### 5. Polynomial Feature Expansion
```python
p = PolynomialFeatures(degree=2)
x_train_poly = p.fit_transform(x_train_pca)
x_test_poly = p.transform(x_test_pca)
```
*Create polynomial features (degree=2) to capture non-linear relationships in the data.*

### 6. Linear Regression with Polynomial Features
```python
model = LinearRegression()
model.fit(x_train_poly, y_train)
y_pred = model.predict(x_test_poly)
r2_score(y_test, y_pred)  # Result: 0.341
```
*Train a linear regression model on polynomial features, achieving an R² score of 0.341.*

### 7. Lasso Regression
```python
La = Lasso(alpha=1)
La.fit(x_train, y_train)
print("test", La.score(x_test, y_test))  # Result: 0.284
print("train", La.score(x_train, y_train))  # Result: 0.290
```
*Implement Lasso regression (L1 regularization) with alpha=1 to prevent overfitting and perform feature selection.*

### 8. Ridge Regression
```python
Ra = Ridge(alpha=100)
Ra.fit(x_train, y_train)
print("test", Ra.score(x_test, y_test))  # Result: 0.580
print("train", Ra.score(x_train, y_train))  # Result: 0.612
```
*Implement Ridge regression (L2 regularization) with a strong regularization parameter (alpha=100).*

### 9. Decision Tree Regressor
```python
clf = DecisionTreeRegressor()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
r2_score(y_test, y_pred)  # Result: 0.616
```
*Use a Decision Tree model which can capture non-linear relationships without feature engineering.*

### 10. Random Forest Regressor
```python
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
r2_score(y_test, y_pred)  # Result: 0.806
```
*Implement an ensemble method with 200 decision trees, achieving the best performance (R² = 0.806).*

### 11. Support Vector Regression
```python
svr = SVR(C=5)
svr.fit(x_train, y_train)
y_pred_svr = svr.predict(x_test)
r2_score(y_test, y_pred_svr)  # Result: 0.069
```
*Test Support Vector Regression with a regularization parameter C=5, which performed poorly on this dataset.*

## 🎯 Key Findings
1. **Random Forest** performed best with an R² score of 0.806, indicating it effectively captured the complex relationships in the data.
2. **Regularization techniques** (Ridge and Lasso) helped prevent overfitting but with varying success.
3. **Feature engineering** (PCA + Polynomial Features) improved linear model performance but couldn't match tree-based methods.
4. **SVR** performed poorly, suggesting the data might not be well-suited for kernel methods without extensive parameter tuning.
