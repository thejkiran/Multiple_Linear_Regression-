#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[109]:


#Read the csv data
df = pd.read_csv("C:/Users/kanch/OneDrive/Desktop/project/Project4_4/insurance.csv")


# In[110]:


x = df.iloc[:, :-1]
y = df.iloc[ :, 6]


# In[111]:


df


# In[112]:


#Convert the columns into categorical columns
Sex = pd.get_dummies(x['sex'], drop_first = False)


# In[113]:


Smoker = pd.get_dummies(x['smoker'], drop_first = False)


# In[114]:


Region = pd.get_dummies(x['region'], drop_first = False)


# In[115]:


#Drop the sex , smoker and region column
x = x.drop('sex', axis =1)
x = x.drop('smoker', axis =1)
x = x.drop('region', axis = 1)


# In[116]:


#concat the dummy variables
x = pd.concat([x,Sex, Smoker, Region], axis = 1)


# In[117]:


x


# In[118]:


#splitting the data into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0 )


# In[119]:


df


# In[120]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[121]:


#predicting the test set results
y_pred = regressor.predict(x_test)


# In[122]:


from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)


# In[123]:


print(score)

