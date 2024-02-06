# %% [markdown]
# # Differences Between Linear Classifier and Linear Regression
# We start with loading a dataset that was created for this discussion and talk a about the differences between linear regression and linear classifier.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
lc2_data = np.genfromtxt('./data/lc2_data.txt', delimiter=None)
X, Y = lc2_data[:, :-1], lc2_data[:, -1]

print(X.shape, Y.shape)
print(X, Y)
f, ax = plt.subplots(1, 2, figsize=(20, 8))

mask = Y == -1

ax[0].scatter(X[mask, 0], X[mask, 1], s=120, color='blue', marker='s', alpha=0.75)
ax[0].scatter(X[~mask, 0], X[~mask, 1], s=340, color='red', marker='*', alpha=0.75)

ax[0].set_xticks(ax[0].get_xticks())
ax[0].set_yticks(ax[0].get_yticks())
ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=25)
ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=25)

ax[1].scatter(X[:, 0], X[:, 1], s=120, color='black', alpha=0.75)

ax[1].set_xticks(ax[1].get_xticks())
ax[1].set_yticks(ax[1].get_yticks())
ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=25)
ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=25)

plt.show()


