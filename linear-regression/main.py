# %% [markdown]
# # Linear Regression
# 
# The code for the linear regression model sits in ml.linear.lnearRegress.

# %%
import numpy as np
import matplotlib.pyplot as plt
import utils as ml
from numpy import atleast_2d
from numpy import asarray

np.random.seed(0)

# %% [markdown]
# ## Simple Stimulation
# 
# Starting with a simple example where the linear regression model has to fit a simple slope line.

# %%
# First we create the "fake data" with xs from 0 to 10 and Y = X + 2
X = np.linspace(0, 10, 50)
print(X.shape)
Y = np.copy(X) + 2
print(Y.shape)

# %%
# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(X, Y, s=80, color='blue', alpha=0.75)

ax.set_xlim(-0.2, 10.2)
ax.set_ylim(1.8, 12.2)
ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

plt.show()    

# %% [markdown]
# Now let's create a test and train data out of it.

# %% [markdown]
# ``` python
# X, Y = ml.shuffleData(X, Y) 
# 
# # ---------------------------------------------------------------------------
# # IndexError                                Traceback (most recent call last)
# # Cell In[14], line 1
# # ----> 1 X, Y = ml.shuffleData(X, Y)
# 
# # File ~/Code/Study/machine-learning-collection/linear-regression/utils/utils.py:163, in shuffleData(X, Y)
# #     160 ny = len(Y)
# #     162 pi = np.random.permutation(nx)
# # --> 163 X = X[pi,:]
# #     165 if ny > 0:
# #     166     assert ny == nx, 'shuffleData: X and Y must have the same length'
# 
# # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
# ```

# %% [markdown]
# If we run the above code, it will have a error.
# 
# This error is a result of some assumptions in the code. All the mltools package code assumes that the X is a 2d array and not 1d. This is a common assumption and it is also used in more popular packages.
# 
# There are many ways to convert from 1d to 2d. The most popular is the atleast_2d

# %%
_ = np.atleast_2d(X).T
print(_.shape)

# %% [markdown]
# Another option is to use reshape. Look at the documentation to see what's the -1 is all about.

# %%
X = X.reshape(-1, 1)
print(X.shape)

# %% [markdown]
# Notice that I transformed it after the atleast2d call. That's because it is common to think of X where the rows are the points and the columns are the dimensions. Please play around with those methods to make sure you understand what it's doing.
# 
# Now let's continue from where we stopped.

# %%
print(X.shape, Y.shape)
X, Y = ml.shuffleData(X, Y)
Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)
print(Xtr.shape, Xte.shape, Ytr.shape, Yte.shape)

# %%
# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')

ax.set_xlim(-0.2, 10.2)
ax.set_ylim(1.8, 12.2)

ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=30, loc=4)

plt.show()    

# %% [markdown]
# Now let's see how we can call the linear regression.

# %%
lr = ml.linear.linearRegress(Xtr, Ytr)

# %% [markdown]
# Boom, that's it. But you should go into the code and make sure you understand how it works. 

# %% [markdown]
# ### Plotting the regression line

# %%
# We start with creating a set of xs on the space we want to predict for.
xs = np.linspace(0, 10, 200)
print(xs.shape)

# Converting to the rate shape
xs = np.atleast_2d(xs).T
print(xs.shape)

# And now the prediction
ys = lr.predict(xs)
print(ys.shape)

# %%
# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')

# Also plotting the regression line
ax.plot(xs, ys, lw=3, color='black', alpha=0.75, label='Prediction')

ax.set_xlim(-0.2, 10.2)
ax.set_ylim(1.8, 12.2)

ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=30, loc=4)

plt.show()    

# %% [markdown]
# We can also print the learned regression object. This will show us the coefficients for each feature. Notice that the regression model added a constant for us.

# %%
print(lr)

# %% [markdown]
# The print above means that the linear regression learned the function Y = 2 + 1 * X.

# %% [markdown]
# ## Real Data
# 
# That was a toy example, let's look at how this is done on real data. 

# %%
path_to_file = 'data/poly_data.txt' 
data = np.genfromtxt(path_to_file, delimiter='\t') # Read data from file 

# %%
# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(data[:, 0], data[:, 1], s=80, color='blue', alpha=0.75)

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-13, 18)

ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

plt.show()    

# %% [markdown]
# Now let's repeate everything on the real data.

# %%
X, Y = np.atleast_2d(data[:, 0]).T, data[:, 1]
print(X.shape, Y.shape)
X, Y = ml.shuffleData(X, Y)
Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)
print(Xtr.shape, Xte.shape, Ytr.shape, Yte.shape)

lr = ml.linear.linearRegress(Xtr, Ytr)

print(lr)

# %%
# Make sure you use the currect space.
xs = np.linspace(0, 4.2, 200)
print(xs.shape)

xs = np.atleast_2d(xs).T

ys = lr.predict(xs)

# %%
# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')

# Also plotting the regression line
ax.plot(xs, ys, lw=3, color='black', alpha=0.75, label='Prediction')

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-13, 18)

ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=30, loc=0)

plt.show()    

# %% [markdown]
# Meh, the predicions don't look that great. Why is that?
# 
# (Because we're fitting Y=X+c line where it's clear that this data comes from a more complex model.)

# %% [markdown]
# So let's fit a more complex model. For that we can use the ml.transform.fpoly method that will convert the features for us.

# %%
degree = 12
XtrP = ml.transforms.fpoly(Xtr, degree, False)

lr = ml.linear.linearRegress(XtrP, Ytr)

print(lr)


# %%
# Make sure you use the currect space.
xs = np.linspace(0, 4.2, 200)
xs = np.atleast_2d(xs).T

# Notice that we have to transform the predicting xs too.
xsP = ml.transforms.fpoly(xs, degree, False)
ys = lr.predict(xsP)

# %%
# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')

# Also plotting the regression line. in the plotting we plot the xs and not the xsP
ax.plot(xs, ys, lw=3, color='black', alpha=0.75, label='Prediction')

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-13, 18)

ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=30, loc=0)

plt.show()    

# %% [markdown]
# Feel free to play around with different degrees and see the differences.

# %% [markdown]
# ## Measuring Prediction Accuracy
# 
# Now you are required to measure the prediction error using MSE and plot it for different degrees.

# %%
def MSE(y_true, y_hat):
    """Mock MSE method.
    
    You'll have to fill it in yourself with the true way of computing the MSE.
    """
    y_true = np.squeeze(y_true)
    y_hat = np.squeeze(y_hat)
    return np.mean((y_true - y_hat)**2)

# %%
# Predicting on the test data - DO NOT FORGET TO TRANSFORM Xte TOO!!!
XteP = ml.transforms.fpoly(Xte, degree, False)
YteHat = lr.predict(XteP)

# %% [markdown]
# Adding the predicted Yhat to the plot. Notice that it sits on the regression line (as expected).

# %%
# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')
ax.scatter(Xte, YteHat, s=80, marker='D', color='forestgreen', alpha=0.75, label='Yhat')

# Also plotting the regression line. in the plotting we plot the xs and not the xsP
ax.plot(xs, ys, lw=3, color='black', alpha=0.75, label='Prediction')

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-13, 18)

ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=20, loc=0)

plt.show()    

# %% [markdown]
# Computing the MSE for the different degrees.

# %%
degrees = np.array([2, 4, 6, 8, 10, 20])
mse_error = np.zeros(degrees.shape[0])

for i, degree in enumerate(degrees):
    XtrP = ml.transforms.fpoly(Xtr, degree, False)

    lr = ml.linear.linearRegress(XtrP, Ytr)
    XteP = ml.transforms.fpoly(Xte, degree, False)
    YteHat = lr.predict(XteP)

    mse_error[i] = MSE(Yte, YteHat)

# %%
f, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plotting a line with markers where there's an actual x value.
# ax.semilogy(degrees, mse_error, lw=4, marker='d', markersize=20, alpha=0.75, label='MSE ERROR')

ax.plot(degrees, mse_error, lw=4, marker='d', markersize=20, alpha=0.75, label='MSE ERROR')

# Setting the X-ticks manually.
ax.set_xticks(np.arange(2, 21, 2))
ax.set_yticks(ax.get_yticks())

ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

ax.legend(fontsize=20, loc=0)

plt.show()

# %% [markdown]
# ## Cross Validation
# 
# Cross-validation works by creating many training/validation splits, called folds, and using all of these splits to assess the “out-of-sample” (validation) performance by averaging them. 

# %% [markdown]
# We provide the function $\bf{ml.crossValidate(Xtr, Ytr, nFolds, iFold)}$ to generate multiple training/validation splits.

# %%
nFolds = 4 

# %%
f, ax = plt.subplots(2, 2, figsize=(20, 20))
ax = ax.flatten()
for iFold in range(nFolds):
    Xti, Xvi, Yti, Yvi = ml.crossValidate(Xtr, Ytr, nFolds, iFold)
    ax[iFold].scatter(Xti, Yti, s=80, color='blue', alpha=0.75, label='Train')
    ax[iFold].scatter(Xvi, Yvi, s=240, marker='*', color='red', alpha=0.75, label='Test')

    ax[iFold].set_xlim(-0.2, 4.3)
    ax[iFold].set_ylim(-13, 18)
    
    ax[iFold].set_xticks(ax[iFold].get_xticks())
    ax[iFold].set_yticks(ax[iFold].get_yticks())
    ax[iFold].set_xticklabels(ax[iFold].get_xticks(), fontsize=25)
    ax[iFold].set_yticklabels(ax[iFold].get_yticks(), fontsize=25)   
    
plt.show()


