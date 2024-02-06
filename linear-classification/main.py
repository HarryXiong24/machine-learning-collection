# %% [markdown]
# # Linear Classification

# %% [markdown]
# It will show how to code a classifier from the ground up. 

# %%
import numpy as np
import matplotlib.pyplot as plt
import utils as ml

np.random.seed(0)

# %%
lc2_data = np.genfromtxt('./data/lc2_data.txt', delimiter=None)
X, Y = lc2_data[:, :-1], lc2_data[:, -1]

print(X.shape, Y.shape)
print(X, Y)

# %% [markdown]
# ## Perceptron Algorithm
# As a simple example we will use the [Perceptron Algorithm](https://en.wikipedia.org/wiki/Perceptron). We will build each part seperately, showing how it works and end by wrapping it all up in a classifier class that can be used with the mltools library. 
# 
# We will use a 2 classes Perceptron with classes $\{-1, 1\}$. You can also see how to use a binary classes $\{0, 1\}$ and in the wiki [page](https://en.wikipedia.org/wiki/Perceptron) you can see a generalization to multiple classes.
# 
# For an illustration of the algorithm you can watch this YouTube [clip](https://www.youtube.com/watch?v=vGwemZhPlsA)

# %% [markdown]
# ### Decision Boundry and Classification
# The Perceptron uses a decision boundary $\theta$ to compute a value for each point. Taking the sign of this value will then give us a class prediction.
# 
# We'll start by computing the decision value for each point $x^j$: $$\theta x^j$$
# 
# As an example, let's choose $j=90$ (the 90th observation in our dataset) and let's define: $$\theta = \left[-6, 0.5, 1\right]$$

# %%
theta = np.array([-6., 0.5, 1.])

# %% [markdown]
# Notice the '.'s. This will make sure it's a float and not integer.

# %%
def add_const(X):
    return np.hstack([np.ones([X.shape[0], 1]), X])

Xconst = add_const(X)
print(X.shape, Xconst.shape)
x_j, y_j = Xconst[90], Y[90]
print(x_j, y_j)

# %% [markdown]
# ### Response Value
# The first step in the preceptron is to compute the response value. It's comptued as the inner (dot) product $\theta x^j$. The simple intuative way to do that is to use a for loop. 

# %%
x_theta = 0
print(theta, x_j)
for i in range(x_j.shape[0]):
    x_theta += x_j[i] * theta[i]
    
print(x_theta)

# %% [markdown]
# This is a VERY inefficient way to do compute a inner product. Luckily for us, numpy has the answer in the form of np.dot().

# %%
print(x_j, theta)
print(np.dot(x_j, theta))

# %% [markdown]
# ### Classification Decision
# Now let's compute the decision classification $T[\theta x^j]$. One option is to use the [np.sign](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sign.html) method. This is not a a good solution because np.sign(0) = 0.
# 
# One solution is to check for 0s explicitly.

# %%
def sign(vals):
    """Returns 1 if val >= 0 else -1"""
    s = np.sign(vals)
    try: # If vals is an array
        s[s == 0] = 1
    except: # If vals is a float
        s = 1 if s == 0 else s
    return s

# %% [markdown]
# ### Predict function
# So now with the the decision value and sign function we can write the predict function

# %%
def predict(x_j, theta):
    """Returns the class prediction of a single point x_j"""
    return sign(np.dot(x_j, theta))

# %%
print(predict(x_j, theta))

# %% [markdown]
# ### Computing the Prediction Error
# Using the predict function, we can now compute the prediction error: $$J^j = (y^j - \hat{y}^j)$$

# %%
def pred_err(X, Y, theta):
    """Predicts that class for X and returns the error rate. """
    Yhat = predict(X, theta)
    return np.mean(Yhat != Y)

# %%
print(pred_err(x_j, y_j, theta))

# %% [markdown]
# ### Learning Update
# Using the error we can now even do the update step in the learning algorithm: $$\theta = \theta + \alpha * (y^j - \hat{y}^j)x^j$$

# %%
a = 0.1
y_hat_j = predict(x_j, theta)
print(theta, y_hat_j, x_j)
print(theta + a * (y_j - y_hat_j) * x_j)

# %% [markdown]
# ### Train method
# Using everything we coded so far, we can fully create the train method

# %%
def train(X, Y, a=0.01, stop_tol=1e-8, max_iter=1000):
    # Start by adding a const
    Xconst = add_const(X)
    
    m, n = Xconst.shape
    
    # Initializing theta
    theta = np.array([-6., 0.5, 1.])
    
    # The update loops
    J_err = [np.inf]
    
    for i in range(1, max_iter + 1):             # Pass through the dataset max_iter times
        
        for j in range(m):                       # Loop through each observation
            x_j, y_j = Xconst[j], Y[j]           # Get observation j
            y_hat_j = predict(x_j, theta)        # Predict using the current theta
            theta += a * (y_j - y_hat_j) * x_j   # Update theta

        # Compute the error on the dataset after each pass
        curr_err = pred_err(Xconst, Y, theta)
        J_err.append(curr_err)

        # Stop if the change in error is small
        if np.abs(J_err[-2] - J_err[-1]) < stop_tol:
            print ('Reached convergance after %d iterations. Prediction error is: %.3f' % (i, J_err[-1]))
            break
        
    return theta

# %%
theta_trained = train(X, Y)

# %% [markdown]
# ## Creating a Perceptron Classifier
# Now let's use all the code that we wrote and create a Python class Perceptron that can plug in to the mltools package.
# 
# In order to do that, the Prceptron class has to inherit the object mltools.base.classifier
# 
# In case you haven't looked at the actual code in the mltools, now will probably be the right time.

# %%
from utils.base import classifier

# %% [markdown]
# In order to crete an object, we'll have to add self to all the methods.

# %%
class Perceptron(classifier):
    def __init__(self, theta=None):
        self.theta = theta
    
    def predict(self, X):
        """Retruns class prediction for either single point or multiple points. """
        # I'm addiing this stuff here so it could work with the plotClassify2D method.
        Xconst = np.atleast_2d(X)
        
        # Making sure it has the const, if not adding it.
        if Xconst.shape[1] == self.theta.shape[0] - 1:
            Xconst = add_const(Xconst)
        
        return self.sign(np.dot(Xconst, self.theta))
                
    def sign(self, vals):
        """A sign version with breaking 0's as +1. """
        return np.sign(vals + 1e-200)
    
    def pred_err(self, X, Y):
        Yhat = self.predict(X)
        return np.mean(Yhat != Y)
    
    def train(self, X, Y, a=0.02, stop_tol=1e-8, max_iter=1000):
        # Start by adding a const
        Xconst = add_const(X)

        m, n = Xconst.shape
        
        # Making sure Theta is inititialized.
        if self.theta is None:
            self.theta = np.random.random(n)

        # The update loops
        J_err = [np.inf]
        for i in range(1, max_iter + 1):
            for j in range(m):
                x_j, y_j = Xconst[j], Y[j]
                y_hat_j = self.predict(x_j)
                self.theta += a * (y_j - y_hat_j) * x_j
                
            curr_err = self.pred_err(Xconst, Y)
            J_err.append(curr_err)

            if np.abs(J_err[-2] - J_err[-1]) < stop_tol:
                print ('Reached convergance after %d iterations. Prediction error is: %.3f' % (i, J_err[-1]))
                break

# %% [markdown]
# ### Creating a model, training and plotting predictions

# %% [markdown]
# First let's create the model with some initialized theta and plot the decision bounderies. For the plotting we can use the mltools plotClassify2D.

# %%
model = Perceptron()
model.theta = np.array([-6., 0.5, 1])

ml.plotClassify2D(model, X, Y)
plt.show()

# %% [markdown]
# Next, let's actually train the model and plot the new decision boundery.

# %%
model.theta = np.array([-6., 0.5, 1])
model.train(X, Y)
ml.plotClassify2D(model, X, Y)

# %% [markdown]
# We found the best classifier!!!


