#!/usr/bin/env python
# coding: utf-8

# ## OASIS Infotech Internship
# ## Intern - Jawahar
# ## Domain - Data Science
# ## Task-1
# ## Iris Flower Classification

# ## Import Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score


# ## Load the dataset & EDA

# In[2]:


data = pd.read_csv("Iris.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


sns.countplot('Species',data=data)


# In[7]:


data =data.drop(['Id'], axis = 1)
data.head()


# In[8]:


data['Species'].value_counts()


# In[9]:


data['Species'].value_counts()


# In[10]:


data['Species'].unique()


# In[11]:


grouped_df= data.groupby('Species')
grouped_df.head()  


# In[12]:


ax = data[data.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                    color='red', label='Iris - Setosa')
data[data.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                color='green', label='Iris - Versicolor', ax=ax)
data[data.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                color='blue', label='Iris - Virginica', ax=ax)
ax.set_title("Scatter Plot")


# In[13]:


ax = data[data.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm', y='PetalWidthCm', 
                                                    color='red', label='Iris - Setosa')
data[data.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm', y='PetalWidthCm', 
                                                color='green', label='Iris - Versicolor', ax=ax)
data[data.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm', y='PetalWidthCm', 
                                                color='blue', label='Iris - Virginica', ax=ax)
ax.set_title("Scatter Plot")


# In[15]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Species'] = encoder.fit_transform(data['Species'])
data


# ## Train_Test_Spilt model

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X= data.drop('Species', axis= 1)
y= data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Using KMeans 

# In[19]:


kmeans= KMeans(n_clusters= 3, random_state= 4)
kmeans.fit(X_train, y_train)


# ## Predictions and Evaluations

# In[20]:


train_labels= kmeans.predict(X_train)
train_labels


# In[21]:


accuracy_score(y_train, train_labels)


# In[22]:


y_pred= kmeans.predict(X_test)
y_pred


# In[23]:


accuracy_score(y_test, y_pred)


# In[24]:


prediction=kmeans.predict(X_test)


# In[25]:


print(classification_report(y_test,prediction))


# In[ ]:




