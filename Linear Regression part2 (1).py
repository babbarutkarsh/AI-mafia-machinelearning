#!/usr/bin/env python
# coding: utf-8

# In[6]:





# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./Train.csv')
print(df.head())

dft=pd.read_csv('Test.csv')

xdata = df.loc[:, ['feature_1','feature_2','feature_3','feature_4','feature_5']]
ydata = df.loc[:, ['target']]

xtest=dft.loc[:, ['feature_1','feature_2','feature_3','feature_4','feature_5']]

X = xdata.values
Y =ydata.values
print(X.shape)
print(Y.shape)

print(X[0])
itr =100


# In[39]:


def hypothesis(x,theta):
      return theta[0]*x[0] + theta[1]*x[1] + theta[2]*x[2] + theta[3]*x[3] + theta[4]*x[4] + theta[5]


def error(X,Y,theta):
    error = 0.0
    m = x.shape[0]
    for i in range(m):
        hx = hypothesis(X[i],theta)
        error += (hx-Y[i])**2        
    return error

def gradient(X,Y,theta):
    
    grad = np.zeros((6,))
    m = X.shape[0]

    for i in range(m):
        hx = hypothesis(X[i],theta)
        grad[5] +=  (hx-Y[i])
        grad[1] += (hx-Y[i])*X[i][1]
        grad[2] += (hx-Y[i])*X[i][2]
        grad[3] += (hx-Y[i])*X[i][3]
        grad[4] += (hx-Y[i])*X[i][4]
        grad[0] += (hx-Y[i])*X[i][0]
        
    return grad
    
def gradientDescent(X,Y,learning_rate=0.0001):
    
    theta = np.array([5.0,2.0,3.0,4.0,5.0,0.0])
    
    itr = 0
    max_itr = 100
    
    error_list = []
    theta_list = []
    
    while(itr<=max_itr):
        grad = gradient(X,Y,theta)
        e = error(X,Y,theta)
        error_list.append(e)
        
        theta_list.append((theta[0],theta[1],theta[2],theta[3],theta[4],theta[5]))
        theta[5] = theta[5] - learning_rate*grad[5]
        theta[1] = theta[1] - learning_rate*grad[1]
        theta[2] = theta[2] - learning_rate*grad[2]
        theta[3] = theta[3] - learning_rate*grad[3]
        theta[4] = theta[4] - learning_rate*grad[4]
        theta[0] = theta[0] - learning_rate*grad[0]

        
        
        itr += 1
        
    
    return theta,error_list,theta_list


# In[40]:


final_theta, error_list,theta_list = gradientDescent(X,Y)


# In[41]:


plt.plot(error_list)
plt.show()
print(final_theta)


# In[42]:


print(xtest)


# In[43]:


print(final_theta)


# In[ ]:





# In[ ]:




