#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Regression

# ## Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# ## Training the Decision Tree Regression model on the whole dataset

# In[4]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)


# ## Predicting a new result

# In[5]:


regressor.predict([[6.5]])


# ## Visualising the Decision Tree Regression results (higher resolution)

# In[12]:


x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff(Decision Tre Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

