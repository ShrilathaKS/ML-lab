#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt 
from sklearn import tree, metrics, model_selection, preprocessing

# load the iris data
df = pd.read_csv('iris.csv')
df.head(5)

df['species_label'],i = pd.factorize(df['species'])
df['species'].unique()
df['species_label'].unique()


# select features
y = df['species_label'] #Dependent feature
X = df[['sepal.length', 'sepal.width']] #Independent features (subset)

# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)


# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtree.fit(X_train, y_train)

# testing the model
y_pred = dtree.predict(X_test)

#acuuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


from sklearn.metrics import  confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[7]:


#Increasing the accuracy of the model by changing the max depth of the tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
dtree.fit(X_train, y_train)

# testing the model
y_pred = dtree.predict(X_test)

#acuuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


from sklearn.metrics import  confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[8]:


#changing the criterion
dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

#acuuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


from sklearn.metrics import  confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[46]:


#Printing the precision, Recall, F1 score for the model
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:




