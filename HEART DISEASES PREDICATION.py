#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[71]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Processing

# In[97]:


# loading the csv data to a Pandas as data frames
heart= pd.read_csv('heart.csv')


# In[98]:


# print first 5 rows of the dataset
heart.head()


# In[74]:


heart.info()


# In[75]:


# print last 5 rows of the dataset
heart.tail()


# In[76]:


# number of rows and columns in the dataset
heart.shape


# In[77]:


# getting some info about the data
heart.info()


# In[78]:


# checking for missing values
heart.isnull().sum()


# In[79]:


# statistical measures about the data
heart.describe()


# In[80]:


# checking the distribution of Target Variable
heart['target'].value_counts()


# 1 --> Defective Heart
# 
# 0 --> Healthy Heart

# Splitting the Features and Target

# In[81]:


X = heart.drop(columns='target', axis=1)
Y = heart['target']


# In[82]:


print(X)


# In[83]:


print(Y)


# Splitting the Data into Training data & Test Data

# In[84]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[85]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[86]:


model = LogisticRegression()


# In[87]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[88]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[89]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[90]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[91]:


print('Accuracy on Test data : ', test_data_accuracy)


# Building a Predictive System

# In[92]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




