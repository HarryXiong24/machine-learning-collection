# %% [markdown]
# # KNN

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# %%
# Get Datasets
iris = load_iris()
print(iris.data, iris.target)

# %%
# Handle Data & Data Split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42, test_size=0.3)

# %%
# Feature Preprocessing
transfer = StandardScaler()
x_train = transfer.fit_transform(X_train)
x_test =  transfer.fit_transform(X_test)

# %%
# Train Model and help to choose best parameter
knn = KNeighborsClassifier(n_neighbors = 5)

param_grid = {"n_neighbors": [1,3,5,7,9]}
knn = GridSearchCV(knn, param_grid=param_grid, cv=10)

knn.fit(x_train, y_train)

# %%
# # Cross Validation
# cross_val_scores = cross_val_score(knn, X_train, y_train, cv=5)
# print(f"Cross-Validation Accuracy Scores: {cross_val_scores}")

# # Get Mean Score
# mean_cv_score = cross_val_scores.mean()
# print(f"Mean Cross-Validation Score: {mean_cv_score}")


# %%
# Evaluate Model

# Output predict value
y_pre = knn.predict(x_test)
print(f'Predict value: {y_pre}\n')
print(f'Predict VS Actual: {y_pre == y_test}\n')

# Output accuracy rate
rate = knn.score(x_test, y_test)
print(f'Accuracy rate: {rate}\n')

# Other
print(f'Best modal: {knn.best_estimator_}\n')
print(f'Best result: {knn.best_score_}\n')
print(f'Best model result: {knn.cv_results_}\n')


