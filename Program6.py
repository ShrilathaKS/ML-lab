#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn import tree, metrics, model_selection, preprocessing 
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


y = df.iloc[:,-1].values
x = df.iloc[:,0:4].values


# In[4]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)


# In[5]:


classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)


# In[6]:


y_predict = classifier.predict(x_test)


# In[7]:


accuracy = metrics.accuracy_score(y_test, y_predict)
accuracy


# In[8]:


from sklearn.metrics import  confusion_matrix
confusion_matrix(y_test, y_predict)


# In[ ]:




