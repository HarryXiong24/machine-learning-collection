#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the text file
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns


# In[3]:


# Use X.shape to get the number of features and the data points. 
# Report both numbers, mentioning which number is which.
data_points, features = X.shape
print("Number of features: ", features)
print("Number of data points: ", data_points)


# In[4]:


# For each feature, plot a histogram ( plt.hist ) of the data values
bins = int(1 + np.log2(data_points)) 

plt.figure(figsize=(15, 5)) 

for i in range(features):
    plt.subplot(1, features, i+1) 
    plt.hist(X[:, i], bins)
    plt.title(f'Feature {i+1}')

plt.tight_layout() 
plt.show()


# In[5]:


# Compute the mean & standard deviation of the data points for each feature ( np.mean , np.std )
means = np.mean(X, axis=0)
standard_deviation = np.std(X, axis=0)

for i in range(4):
    print("Feature ", i+1)
    print("Mean: ", means[i])
    print("Standard Deviation: ", standard_deviation[i])


# In[6]:


# For each pair of features (1,2), (1,3), and (1,4), plot a scatterplot (see plt.plot or plt.scatter ) of the feature values, colored according to their target value (class). (For example, plot all data points with y = 0 as blue, y = 1 as green, etc.) 

colors = []
for y in Y:
    if y == 0:
        colors.append('red')
    elif y == 1:
        colors.append('blue')
    else:
        colors.append('green')
        
plt.figure(figsize=(15, 5)) 

for i in range(1, features):
    plt.subplot(1, features, i+1) 
    plt.scatter(X[:,0], X[:,i], c=colors)
    plt.title(f'Feature (1, {i+1})')
    
plt.tight_layout() 
plt.show() 

