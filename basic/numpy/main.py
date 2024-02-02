#!/usr/bin/env python
# coding: utf-8

# # Numpy

# ## Import

# In[109]:


import numpy as np


# ## Create & Type

# In[110]:


# Create a numpy array
np1 = np.array([0, 1, 2, 3, 4, 5, 6])
print(np1)
# Print the array
print(np1.shape)

# Create a numpy array
my_list = [0, 1, 2, 3, 4, 5, 6]
np2 = np.array(my_list)

print(np2.dtype)


# ## Common ways to create array

# In[111]:


print(np.arange(10))
print(np.arange(0, 10, 2))

print(np.zeros(10))
print(np.zeros((2, 10)))

print(np.ones(10))
print(np.ones((2, 10)))

print(np.full((10), 6))
print(np.full((2, 10), 6))

print(np.eye(4))
print(np.eye(4, 4))


# In[112]:


# random
np1 = np.random.random((2, 10))
print(np1)

np2 = np.random.randint(0, 10, (2, 10))
print(np2)


# ## Dimension

# In[113]:


# ndim & shape & size & itemsize
np1 = np.random.randint(0, 10, 10)
print(np1)
print(np1.ndim)
print(np1.shape)
print(np1.size)
print(np1.itemsize)

print('----------------')
np2 = np.random.randint(0, 10, (2, 10))
print(np2)
print(np2.ndim)
print(np2.shape)
print(np2.size)
print(np2.itemsize)

print('----------------')
np3 = np.random.randint(0, 10, (2, 3, 4))
print(np3)
print(np3.ndim)
print(np3.shape)
print(np3.size)
print(np3.itemsize)


# In[114]:


# reshape
np1 = np.random.randint(0, 10, 20)
print(np1, np1.shape)

print('----------------')
np2 = np1.reshape(2, 10)
print(np2, np2.shape)

print('----------------')
np3 = np1.reshape(2, 2, 5)
print(np3, np3.shape)

# -1 means let numpy figure out the dimension size in that dimension
print('----------------')
np4 = np3.reshape((-1, 1))
print(np4, np4.shape)

# -1 means let numpy figure out the dimension size in that dimension
print('----------------')
np5 = np3.reshape((1, -1))
print(np5, np5.shape)

print('----------------')
np6 = np3.flatten() # flatten() always returns a copy
print(np6, np6.shape)

print('----------------')
np7 = np3.ravel() # ravel() returns a view if possible
print(np7, np7.shape)


# ## Slice

# In[115]:


# Slice in one dimension
np1 = np.arange(10)
print(np1)

print(np1[4])
print(np1[0:4])
print(np1[0:10:2])


# In[116]:


# Slice in two dimension
np2 = np.arange(24).reshape(4, 6)
print(np2)

print('----------------')
print(np2[0])
print(np2[0, 0])

print('----------------')
print(np2[0:1, :])
print(np2[:, 0])
print(np2[0:2, 0:3])


# ## Bool Index

# In[117]:


np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
np2 = np1 < 10
print(np2)

print('----------------')
np3 = np1[np1 < 10]
print(np3)

print('----------------')
np4 = np1[np1 % 2 == 0]
print(np4)

print('----------------')
np5 = np1[(np1 < 5) & (np1 > 10)]
print(np5)

print('----------------')
np6 = np1[(np1 < 5) | (np1 > 10)]
print(np6)


# ## Value Update

# In[118]:


np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
np1[0] = 100
print(np1)

print('----------------')
np1[np1 < 10] = 0
print(np1)

print('----------------')
np2 = np.where(np1 > 20, 0, 1)
print(np2)


# ## Broadcast

# In[119]:


# array and number
np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
np2 = np1 + 10
print(np2)

print('----------------')
np3 = np1 * 10
print(np3)


# In[120]:


# array and array
np1 = np.arange(24).reshape(4, 6)
print(np1)

# shape must be the same
print('----------------')
np2 = np.arange(24).reshape(4, 6) 
print(np1 + np2) 

# row is the same
print('----------------')
np3 = np.arange(4).reshape(-1, 1)
print(np3)
print(np1 + np3)

# column is the same
print('----------------')
np4 = np.arange(6)
print(np4)
print(np1 + np4)


# ## Arrays Concatenate

# In[121]:


# row
np1 = np.arange(24).reshape(4, 6)
print(np1)
np2 = np.arange(12).reshape(-1, 6)
print(np2)

np3 = np.vstack((np1, np2))
print(np3)

# column
print('----------------')
np4 = np.arange(24).reshape(4, 6)
print(np4)
np5 = np.arange(12).reshape(4, -1)
print(np5)

np6 = np.hstack((np4, np5))
print(np6)


# In[122]:


# concatenate

# column
np1 = np.arange(24).reshape(4, 6)
print(np1)
np2 = np.arange(12).reshape(4, -1)
print(np2)

np3 = np.concatenate((np1, np2), axis=1)
print(np3)

# row
print('----------------')
np4 = np.arange(24).reshape(4, 6)
print(np4)
np5 = np.arange(12).reshape(-1, 6)
print(np5)

np6 = np.concatenate((np4, np5), axis=0)
print(np6)

# flatten
print('----------------')
np7 = np.concatenate((np4, np5), axis=None)
print(np7)


# ## Array Split

# In[123]:


np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
v1, v2 = np.vsplit(np1, 2)
print(v1, v2)

print('----------------')
v1, v2, v3, v4 = np.vsplit(np1, 4)
print(v1, v2, v3, v4)

print('----------------')
h1, h2 = np.hsplit(np1, 2)
print(h1, h2)

print('----------------')
h1, h2, h3 = np.hsplit(np1, 3)
print(h1, h2, h3)


# In[124]:


# array_split
np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
v1, v2 = np.array_split(np1, 2, axis=0)
print(v1, v2)

print('----------------')
v1, v2, v3 = np.array_split(np1, 3, axis=1)
print(v1, v2, v3)


# ## Array Transpose

# In[125]:


np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
np2 = np1.T
print(np2)

print('----------------')
np3 = np1.transpose()
print(np3)

print('----------------')
np4 = np1.dot(np2)
print(np4)


# ## Array Copy

# In[126]:


# no copy
np1 = np.arange(24).reshape(4, 6)
print(np1)
np2 = np1
print(np2)

print('----------------')
np2[0, 0] = 1
print(np1, np2)


# In[127]:


# shallow copy
np1 = np.arange(24).reshape(4, 6)
print(np1)
np2 = np1.view()
print(np2)

print('----------------')
np2[0, 0] = 1
print(np1, np2)


# In[128]:


# deep copy
np1 = np.arange(24).reshape(4, 6)
print(np1)
np2 = np1.copy()
print(np2)

print('----------------')
np2[0, 0] = 1
print(np1, np2)


# ## CSV

# In[129]:


np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
np.savetxt('np1.csv', np1, fmt='%d', delimiter=',', header='column1, column2, column3, column4, column5, column6', comments='') 


# In[130]:


file = np.loadtxt('np1.csv', dtype=np.int32, delimiter=',', skiprows=1)
print(file)


# In[131]:


np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
np.save('np1', np1) 
np.savez('np1', np1) 


# In[132]:


np1 = np.load('np1.npy')
print(np1)


# ## Handle Empty Value

# In[133]:


np1 = np.arange(24).reshape(4, 6);
print(np1)

print('----------------')
np1 = np1.astype(float)
print(np1[0, 1])
np1[0, 1] = np.NaN

print(np1[0,1] == np.NaN) # False, because NaN is not equal to anything, including itself
print(np.isnan(np1[0,1]))
print(np1[0,1] * 2) # NaN, because any operation with NaN will result in NaN


# In[134]:


# When reading from a file, you can use the following code to convert empty strings to NaN
scores = np.genfromtxt('scores.csv', delimiter=',', missing_values='', filling_values='nan')
print(scores)
scores = scores.astype(float)


# ## Random Module

# In[191]:


np.random.seed(1) # if you want the same random number, you can use the same seed
np1 = np.random.rand()
print(np1)

print('----------------')
np2 = np.random.rand(2, 3)
print(np2)

print('----------------')
np3 = np.random.randn(2, 3) # apply normal distribution to the random number
print(np3)

print('----------------')
np4 = np.random.randint(0, 10, (2, 3))
print(np4)

print('----------------')
data = [2, 4, 6, 8, 10]
np5 = np.random.choice(data, (2, 3)) # randomly select from the data
print(np5)

print('----------------')
np6 = np.arange(10)
print(np6)
np.random.shuffle(np6) # shuffle the data
print(np6)


# ## Axis

# In[198]:


np1 = np.arange(24).reshape(4, 6)
print(np1)

print('----------------')
np2 = np1.sum(axis=0)
print(np2)
np3 = np1.sum(axis=1)
print(np3)

print('----------------')
np4 = np1.max(axis=0)
print(np4)
np5 = np1.max(axis=1)
print(np5)

print('----------------')
np6 = np.delete(np1, 0, axis=0)
print(np6)

np7 = np.delete(np1, 1, axis=1)
print(np7)


# ## Common Function

# [Common Function](https://juejin.cn/post/7001376518555303950?searchId=20240202025842E24ADE3219F7076CC15E#heading-13)

# In[207]:


np1 = np.random.uniform(-10, 10, (2, 4))
print(np1)

print('----------------')
np2 = np.abs(np1)
print(np2)

print('----------------')
np3 = np.sqrt(np2)
print(np3)

print('----------------')
np4 = np.square(np3)
print(np4)

print('----------------')
np5 = np.log(np2)
print(np5)


# ![](./two.png)

# ![](./aggregate.png)

# In[209]:


# boolean
np1 = np.random.randint(0, 10, (2, 4))
print(np1)

print('----------------')
res1 = np.all(np1 > 0)
print(res1)

print('----------------')
res2 = np.any(np1 > 0)
print(res2)


# In[225]:


# sort
np.random.seed(1)
np1 = np.random.randint(0, 10, (2, 4))
print(np1)

print('----------------')
np2 = np.sort(np1, axis=0) # return the sorted array
print(np2)

print('----------------')
np3 = np.sort(np1, axis=1) # return the sorted array
print(np3)

print('----------------')
np4 = np.argsort(np1, axis=0) # return the index of the sorted array
print(np4)
np5 = np.argsort(np1, axis=1) # return the index of the sorted array
print(np5)

print('----------------')
np6 = -np.sort(-np1, axis=1) # return the sorted array in descending order
print(np6)


# In[ ]:




