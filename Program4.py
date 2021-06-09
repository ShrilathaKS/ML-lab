#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import pylab as pl
from sklearn import linear_model,metrics
from sklearn.metrics import mean_squared_error


df = pd.read_csv("FuelConsumption.csv")
df.head()
cdf  =  df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()



msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test =  cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
plt.show()


reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])


reg.fit(train_x,train_y)


print('Coefficients of the model:', reg.coef_)

y_pred = reg.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

#printing intercept 
print('Intercept is :',reg.intercept_)

#Plotting the graph between predicted and actual values
plt.scatter(y_test,y_pred)

#printing error
b = mean_squared_error(y_test,y_pred)
print('Mean Squared error',b)

#printing accuracy of the model
print("Accuracy score : %2f" %reg.score(x_test,y_test))


# In[ ]:




