#!/usr/bin/env python
# coding: utf-8

# In[7]:



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')
df.head()
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
plt.scatter(x,y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

y_pred

viz_train = plt
viz_train.scatter(x_train, y_train, color='red')
viz_train.plot(x_train, reg.predict(x_train), color='blue')
viz_train.title('Salary vs Experience(training data)')
viz_train.xlabel('Years of Experience')
viz_train.ylabel('Salary')


# In[ ]:




