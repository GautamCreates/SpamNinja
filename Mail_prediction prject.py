#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# In[3]:


df1=pd.read_csv(r"C:\Users\Akash\Downloads\archive (1)\mail_data.csv")
df1


# In[4]:


df1.shape


# In[5]:


df1.head()


# In[6]:


df=df1.where((pd.notnull(df1)),'')


# In[7]:


df.loc[df['Category']=='spam','Category']=0
df


# In[8]:


df.loc[df['Category']=='ham','Category']=1
df


# In[9]:


X=df.Message
X


# In[10]:


Y=df.iloc[:,[0]]
Y


# In[11]:


X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.3)


# In[12]:


X_train.shape


# In[13]:


x_test.shape


# In[14]:


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
x_test_features=feature_extraction.transform(x_test)


# In[16]:


print(X_train_features)


# In[17]:


Y_train=Y_train.astype('int')
y_test=y_test.astype('int')


# In[18]:


Y_train


# In[19]:


print(X_train_features)


# In[20]:


print(x_test_features)


# In[21]:


model=LogisticRegression()


# In[22]:


model.fit(X_train_features,Y_train)


# In[23]:


prediction_on_train_data=model.predict(X_train_features)
accuracy_on_train_data=accuracy_score(Y_train,prediction_on_train_data)
print("accuracy_on_train_data",accuracy_on_train_data)


# In[24]:


prediction_on_test_data=model.predict(x_test_features)
accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)
print("accuracy_on_test_data",accuracy_on_test_data)


# In[ ]:





# In[ ]:




