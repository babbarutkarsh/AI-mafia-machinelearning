#!/usr/bin/env python
# coding: utf-8

# Assignment 2
# Linear regression 
#     

# In[80]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[107]:


#dfx the training data
dfx=pd.read_csv('Linear_X_Train.csv')
dfy=pd.read_csv('Linear_Y_Train.csv')
dft=pd.read_csv('Linear_X_Test.csv')


# In[82]:


#converting the pandas read file to numpy
x=dfx.values
y=dfy.values
xtest=dft.values


# In[83]:


print(x)#its a 2D matrix with one column and multiple rows


# In[84]:


print(y)


# In[85]:


print(x.shape)
print(y.shape)


# In[86]:


# now we are going to reshape the data as 
x=x.reshape((-1,))
#this function reshapes rows 
print(x)


# In[87]:


y=y.reshape((-1,))
#this function reshapes cols
print(y)


# In[88]:


plt.scatter(x,y)
plt.show()


# In[89]:


#normalizing the dataset
x=(x-x.mean())/x.std()
plt.scatter(x,y)
plt.show()


# In[90]:


#thus the x values are shifted across origin
#implementing the gradient descend algo
#steps 
#1. start with random theta, 2. repeat till convergence 3.update the theta according to the rule
def hypothesis(x,theta):
    return theta[0]+theta[1]*x#nothing but y=mx+c here c=theta[0] while m is theta[1]


# In[91]:


#finding the error
def error(x,y,theta):
    err=0
    for i in range(x.shape[0]):
        hx=hypothesis(x[i],theta)
        err+=(hx-y[i])**2
    return err


# In[92]:


#getting the gradient function(del(g)/del(theta))
def gradient(x,y,theta):
    grad=np.zeros((2,))
    for i in range(x.shape[0]):
    #x.shape[0] is basically giving me 99 i.e the number of elements in the matrix
        hx=hypothesis(x[i],theta)
        grad[0]+=(hx-y[i])
        grad[1]+=(hx-y[i])*x[i]
    return grad


# In[97]:


#writing the gradient descend function
def gradientdescent(x,y,learning_rate=0.0001):
    #1.intitialize theta
    theta=np.array([4.0,0.0])
    itr=0
    max_itr=100
    
    error_list=[]
    theta_list=[] 
    
    while(itr<=max_itr):
        grad=gradient(x,y,theta)
        e=error(x,y,theta)
        error_list.append(e)
        theta_list.append((theta[0],theta[1]))
        theta[0]=theta[0]-learning_rate*grad[0]
        theta[1]=theta[1]-learning_rate*grad[1]
        itr+=1
    #calculate the error
    
    return theta,error_list,theta_list


# In[98]:


final_theta, error_list, theta_list=gradientdescent(x,y)#the final theta


# In[99]:


final_theta


# In[100]:


plt.plot(error_list)


# In[101]:


print(final_theta)


# In[ ]:





# In[114]:


xtest


# In[115]:


plt.scatter(x,y,label="training data")
plt.plot(xtest,hypothesis(xtest,final_theta),color='red',label="prediction")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




