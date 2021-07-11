#!/usr/bin/env python
# coding: utf-8

# In[55]:


from sklearn.datasets import load_boston


# In[40]:


boston= load_boston()


# In[41]:


X= boston.data
y= boston.target


# In[42]:


print(X.shape)


# In[43]:


print(y.shape)


# In[44]:


print(boston.feature_names)


# In[45]:


print(boston.DESCR)


# In[46]:


#converting data into a data frame 
import pandas as pd
df= pd.DataFrame(X)
df.columns = boston.feature_names
df.head()


# In[47]:


df.describe()


# # feature engineering 

# In[48]:


#adding target variable to the the set
df['MEDV'] = boston.target


# In[49]:


#counting the misiing values for each feature
df.isnull().sum()


# In[50]:


#plotting the distribution of target variable MEDV 
#using distplot function from seaborn library

import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(df['MEDV'], bins=30)
plt.show()


# In[51]:


#creating the corelation matrix
corr_matrix = df.corr().round(2)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(data=corr_matrix, annot=True)


# In[52]:


#seeing relation of LSTAT and RM with MEDV

plt.figure(figsize=(20,5))

features = ['LSTAT', 'RM']
target = df['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = df[col]
    y = target
    plt.scatter(x, y, marker = '*')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# In[36]:


#normalisation

import numpy as np
u= np. mean(X, axis=0)
std= np.std(X,axis=0)
print(u)


# In[37]:


X=(X-u)/std
pd.DataFrame(X[:5,:]).head()


# In[38]:


# Plot Y vs any feature
plt.style.use('seaborn')
plt.scatter(X[:,5],y)
plt.show()


# # linear regression 

# In[19]:


#matrix update
ones= np.ones((X.shape[0],1))
print(ones[:5])
X=np.hstack((ones,X))
print(X.shape)


# In[20]:


#hypothesis function
# X - Matrix ( m x n)
# x - Vector (Single Example with n features)

def hypothesis(x,theta):
    y_ = 0.0
    n = x.shape[0]
    for i in range(n):
        y_  += (theta[i]*x[i])
    return y_

def error(X,y,theta):
    e = 0.0
    m = X.shape[0]
    
    for i in range(m):
        y_ = hypothesis(X[i],theta)
        e += (y[i] - y_)**2
        
    return e/m

def gradient(X,y,theta):
    m,n = X.shape
    
    grad = np.zeros((n,))
    
    # for all values of j
    for j in range(n):
        #sum over all examples
        for i in range(m):
            y_ = hypothesis(X[i],theta)
            grad[j] += (y_ - y[i])*X[i][j]
    # Out of the loops
    return grad/m
  
def gradient_descent(X,y,learning_rate=0.1,max_epochs=300):
    m,n = X.shape
    theta = np.zeros((n,))
    error_list = []
    
    for i in range(max_epochs):
        e = error(X,y,theta)
        error_list.append(e)
        
        # Gradient Descent
        grad = gradient(X,y,theta)
        for j in range(n):
            theta[j] = theta[j] - learning_rate*grad[j]
        
    return theta,error_list


# In[21]:


import time
start = time.time()
theta,error_list = gradient_descent(X,y)
end = time.time()
lrt =end-start
print("Time taken is ", lrt)


# # optimisation of linear regression

# In[22]:


#X is Matrix (mxn)
#x is vector (Single example with n features)

def opthypothesis(X,theta):
    return np.dot(X,theta)

def opterror(X,Y,theta):
    e = 0.0
    Y_ = opthypothesis(X,theta)
    e = np.sum((Y-Y_)**2)
    m = X.shape[0]
    return e/m

def optgradient(X,Y,theta):
    Y_ = opthypothesis(X,theta)
    grad = np.dot(X.T,(Y_-Y))
    m = X.shape[0]
    return grad/m

def optgradient_descent(X,Y,learning_rate = 0.1, max_iters=300):
    n=X.shape[1]
    theta = np.zeros((n,))
    error_list = [0]
    
    for i in range(max_iters):
        e = opterror(X,Y,theta)
        error_list.append(e)
        
        #Gradient descent
        grad = optgradient(X,Y,theta)
        theta = theta - learning_rate*grad

    return theta,error_list        


# In[23]:


import time
optstart = time.time()
theta,error_list = optgradient_descent(X,y)
optend = time.time()
optlrt =optend-optstart
print("Time taken is ", optlrt)


# In[24]:


print(theta)


# In[25]:


plt.plot(error_list)
plt.show()


# In[26]:


#predictions
y_ = []
m=X.shape[0]
for i in range(m):
    pred = hypothesis(X[i],theta)
    y_.append(pred)
y_ = np.array(y_)


# In[27]:


def r2_score(y,y_):
    num = np.sum((y-y_)**2)
    denom = np.sum((y- y.mean())**2)
    score = (1- num/denom)
    return score*100


# In[28]:


# SCORE
r2_score(y,y_)


# In[29]:


from sklearn import metrics
print('RMSE', np.sqrt(metrics.mean_squared_error(y, y_)))


# In[30]:


df1 = pd.DataFrame({'Actual': y, 'Prediction':y_})
df1.head(20)


# In[31]:


plt.figure(figsize=(16,8))
plt.plot(y,label ='Test',color='yellow')
plt.plot(y_, label = 'predict')
plt.show()


# In[32]:


plt.figure(figsize=(15,8))
plt.scatter(y,y_)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[33]:


df1.head(20).plot(kind = 'bar')


# # Gradient boosting regression
# 

# In[46]:


from sklearn import ensemble
from sklearn.utils import shuffle


# In[47]:


params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.05, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X, y)


# In[48]:


import time
start = time.time()
clf_pred=clf.predict(X)
clf_pred= clf_pred.reshape(-1,1)
end = time.time()
gbt=end-start
print("Time taken is ", gbt)


# In[49]:


print('MAE:', metrics.mean_absolute_error(y, clf_pred))
print('MSE:', metrics.mean_squared_error(y, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y, clf_pred)))


# In[50]:


plt.figure(figsize=(12,6))
plt.scatter(y,clf_pred, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[51]:


plt.figure(figsize=(12,6))
plt.plot(y,label ='Test')
plt.plot(clf_pred, label = 'predict')
plt.show()


# # Random forest regression

# In[56]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)
rfr.fit(X, y)


# In[59]:


import time
start = time.time()
rfr_pred= rfr.predict(X)
print (rfr_pred.shape)
rfr_pred = rfr_pred.reshape(-1,1)
print (rfr_pred.shape)
end = time.time()
rft= end-start
print("Time taken is ", rft)


# In[54]:


import sklearn.metrics as metrics


# In[55]:


print('MAE:', metrics.mean_absolute_error(y, rfr_pred))
print('MSE:', metrics.mean_squared_error(y, rfr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y, rfr_pred)))


# In[56]:


plt.figure(figsize=(12,6))
plt.scatter(y,rfr_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[62]:


plt.figure(figsize=(16,6))
plt.plot(y,label ='Test')
plt.plot(rfr_pred, label = 'predict',color='green')
plt.show()


# # comparision

# In[58]:


plt.figure(figsize=(16,8))
plt.plot(y,label ='Test',color='yellow')
plt.plot(rfr_pred, label = 'random forrest predict',color='blue')
plt.plot(clf_pred, label = 'gradient boost predict',color='red')
plt.plot(y_, label = 'predict',color='green')
plt.show()


# In[59]:


plt.figure(figsize=(15,10))

plt.scatter(y,y_)
plt.scatter(y,clf_pred, c= 'yellow')
plt.scatter(y,rfr_pred, c='brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[72]:


RMSE_gbr = np.sqrt(metrics.mean_squared_error(y, clf_pred))
RMSE_rfr = np.sqrt(metrics.mean_squared_error(y, rfr_pred))
RMSE_lr = np.sqrt(metrics.mean_squared_error(y, y_))

dict = {'RMSE in Linear Regression':RMSE_lr, 'RMSE in Gradient Boosting Regression': RMSE_gbr, 
                    'RMSE in Random Forest Regression':RMSE_rfr}

df3 = pd.DataFrame([dict])
df3.head()


# In[73]:



df3.plot(kind = 'bar')


# In[71]:


time= np.array( [['optimised linear regression',optlrt],['gradient boosting',gbt],['random forrest',rft]])
time= pd.DataFrame(time)
header=np.array(['regression type', 'time taken'])
time.columns=header
time.head()


# In[70]:


xtime= {'Linear Regression':optlrt, 'Gradient Boosting Regression': gbt, 
                   'Random Forest Regression':rft}

df4 = pd.DataFrame([xtime])


df4.plot(kind = 'bar')

