# %% [markdown]
# # Naive Bayes

# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# %%
# Get Datasets
digits = load_digits()
X, y = digits.data, digits.target
print(X.shape[0], y.shape[0])

# %%
# Handle Data & Data Split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=42, test_size=0.3)

# %%
# Train Model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# %%
# Cross Validation
cross_val_scores = cross_val_score(gnb, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy Scores: {cross_val_scores}")

# Get mean value
mean_cv_score = cross_val_scores.mean()
print(f"Mean Cross-Validation Score: {mean_cv_score}")


# %%
# Evaluate Model

# predict
y_pred = gnb.predict(X_test)
print("y_predict: ", y_pred)

# show predict probability
y_predict_proba = gnb.predict_proba(X_test)
print("y_predict_proba: ", y_predict_proba)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# score
score = gnb.score(X_test, y_test)
print("score: ", score)


