#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn import metrics

df = pd.read_csv('iris2.csv')
df.head()

plt.scatter(df['petal.length'],df['petal.width'])
plt.xlabel('length')
plt.ylabel('width')

km = KMeans(n_clusters=3)
yp = km.fit_predict(df)

df['cluster'] = yp
df.head(2)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['petal.length'],df1['petal.width'],color='blue')
plt.scatter(df2['petal.length'],df2['petal.width'],color='green')
plt.scatter(df3['petal.length'],df3['petal.width'],color='red')

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse) 


# In[9]:


#Printing the precision, Recall, F1 score for the model
from sklearn.metrics import classification_report

print(classification_report(df,k_rng))


# In[ ]:




