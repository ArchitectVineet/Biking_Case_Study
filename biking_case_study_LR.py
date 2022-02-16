#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Required Libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


#Reading the data
df=pd.read_csv('day.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# No null values were detected in the dataset.

# ### Dropping all the columns which do not seem to be contributing 

# In[10]:


df.drop(['instant','dteday','casual','registered','holiday'],axis=1,inplace=True)
df.head()


# ### Categorical columns 

# In[11]:


df['season'] = df['season'].replace([1,2,3,4],['spring','summer','fall','winter'])


# In[12]:


df['yr']=df['yr'].replace([0,1],['2018','2019'])


# In[13]:


df['mnth']=df['mnth'].replace([1,2,3,4,5,6,7,8,9,10,11,12],['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


# In[14]:


df['weekday']=df['weekday'].replace([0,1,2,3,4,5,6],['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])


# In[15]:


df['weathersit']=df['weathersit'].replace([1,2,3,4],['Clear','Mist','Light Snow','Heavy Rain'])
df.sample(10)


# In[16]:


df1= df.groupby(df.yr)['cnt'].sum()


# In[17]:


df1.plot.bar()


# In[18]:


df1= df[['yr','season','cnt']]
df1


# In[19]:


plt.figure(figsize=(10, 6))
sb.barplot(x="yr", hue="season", y="cnt", data=df,ci=None)
plt.show()


# In[20]:


sb.barplot('mnth','cnt',hue='yr',data=df,ci=None)


# In[21]:


sb.barplot(x="weekday", hue="yr", y="cnt", data=df,ci=None)
plt.show()


# In[22]:


sb.pairplot(df)
plt.show()


# In[23]:


sb.heatmap(df.corr())

sb.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[24]:


plt.scatter('temp','cnt',data=df)


# In[25]:


df.drop(['atemp'],axis=1,inplace= True)


# In[26]:


df.corr()


# In[27]:


season_dist= pd.get_dummies(df['season'])


# In[28]:


status = pd.get_dummies(df['yr'])


# In[29]:


situation=pd.get_dummies(df['weathersit'])


# In[30]:


df= pd.concat([df,season_dist,status,situation],axis=1)


# In[31]:


df.drop(['mnth','weathersit','yr','season','weekday'],axis= 1,inplace=True)


# In[33]:


from sklearn.preprocessing import MinMaxScaler


# In[34]:


scaler = MinMaxScaler()


# In[35]:


scale_var=['temp','hum','windspeed','cnt']
df[scale_var]= scaler.fit_transform(df[scale_var])


# In[36]:


df.head()


# In[37]:


df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[38]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sb.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[39]:


y_train = df_train.pop('cnt')
X_train = df_train


# In[40]:


import statsmodels.api as sm

# Add a constant
X_train_lm = sm.add_constant(X_train[['temp']])

# Create a first fitted model
lr = sm.OLS(y_train, X_train_lm).fit()


# In[41]:


# Check the parameters obtained

lr.params


# In[42]:


# Let's visualise the data with a scatter plot and the fitted regression line
plt.scatter(X_train_lm.iloc[:, 1], y_train)
plt.plot(X_train_lm.iloc[:, 1], 0.169798 + 0.639952*X_train_lm.iloc[:, 1], 'r')
plt.show()


# In[43]:


print(lr.summary())


# ### Adding another variable 'workingday'

# In[44]:


# Assign all the feature variables to X
X_train_lm = X_train[['temp', '2019']]


# In[45]:


# Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train_lm)

lr = sm.OLS(y_train, X_train_lm).fit()

lr.params


# In[46]:


# Check the summary
print(lr.summary())


# In[47]:


X_train_lm= X_train[['temp','2019','fall']]


# In[48]:


# Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train_lm)

lr = sm.OLS(y_train, X_train_lm).fit()

lr.params


# In[49]:


# Check the summary
print(lr.summary())


# In[ ]:


df.columns


# In[50]:


#Build a linear model for all variables

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)

lr_1 = sm.OLS(y_train, X_train_lm).fit()

lr_1.params


# In[51]:


print(lr_1.summary())


# In[52]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[53]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X= df.drop(['cnt'],axis=1)
y=df['cnt']
X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.3, random_state= 42)


# In[57]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[58]:


scale_var=['temp','hum','windspeed','cnt']
df[scale_var]= scaler.fit_transform(df[scale_var])
df


# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[65]:


print('Intercept: \n', lm.intercept_)
print('Coefficients: \n', lm.coef_)


# In[66]:


y_pred = lm.predict(X_test)


# In[67]:


y_pred


# In[68]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# In[69]:


# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_pred)
print('r2 socre is ',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_pred))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:




