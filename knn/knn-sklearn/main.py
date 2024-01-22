#!/usr/bin/env python
# coding: utf-8

# # Example

# In[4]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. Get Datasets
iris = load_iris()
print(iris.data, iris.target)

# 2. Handle Data & Data Split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22, test_size=0.2)

# 3. Feature Preprocessing
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test =  transfer.fit_transform(x_test)

# 4. Train Model
estimator = KNeighborsClassifier(n_neighbors = 5)
estimator.fit(x_train, y_train)

# 5. Evaluate Model
y_pre = estimator.predict(x_test)
print(f'Predict value: {y_pre}')
print(f'Predict VS Actual: {y_pre == y_test}')

rate = estimator.score(x_test, y_test)
print(f'Accuracy rate: {rate}')


# In[10]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. Get Datasets
iris = load_iris()
print(iris.data, iris.target)

# 2. Handle Data & Data Split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22, test_size=0.2)

# 3. Feature Preprocessing
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test =  transfer.fit_transform(x_test)

# 4. Train Model
estimator = KNeighborsClassifier(n_neighbors = 5)

# Cross Validation
param_grid = {"n_neighbors": [1,3,5,7,9]}
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=10)

estimator.fit(x_train, y_train)

# 5. Evaluate Model

# Output predict value
y_pre = estimator.predict(x_test)
print(f'Predict value: {y_pre}\n')
print(f'Predict VS Actual: {y_pre == y_test}\n')

# Output accuracy rate
rate = estimator.score(x_test, y_test)
print(f'Accuracy rate: {rate}\n')

# Other
print(f'Best modal: {estimator.best_estimator_}\n')
print(f'Best result: {estimator.best_score_}\n')
print(f'Best model result: {estimator.cv_results_}\n')

