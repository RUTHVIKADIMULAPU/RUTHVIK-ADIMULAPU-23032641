#!/usr/bin/env python
# coding: utf-8

# # DIABETES DATASET

# In[42]:


import pandas as pd
import numpy as np


# The datasets consists of several medical predictor variables and one target variable, Outcome.

# Pregnancies :- Number of times pregnant
# 
# Glucose:- Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# BloodPressure:- Diastolic blood pressure
# 
# SkinThickness:- Triceps skin fold thickness
# 
# Insulin:- 2-Hour serum insulin
# 
# BMI:- Body mass index
# 
# DiabetesPedigreeFunction:- Diabetes pedigree function
# 
# Age:-Age in years
# 
# Outcome:- Class variable (0 or 1)

# In[43]:


my_data=pd.read_csv('diabetes.csv')


# In[44]:


my_data


# In[45]:


my_data.shape


# In[46]:


types=my_data.dtypes
types


# In[47]:


#coloumns
my_data.columns


# In[48]:


#Top 5 rows in dataset
my_data.head()


# MISSING VALUES

# In[49]:


my_data.isnull().sum()


# In[50]:


my_data.info()


# In[51]:


#describing the data
my_data.describe()


# In[52]:


my_data.groupby('Outcome').size()


# In[53]:


#Mean
my_data.groupby('Outcome').mean()


# In[54]:


#Median
my_data.groupby('Outcome').median()


# In[55]:


#Standard deviation
my_data.groupby('Outcome').std()


# In[56]:


#skew calculation
my_data.groupby('Outcome').skew()


# In[57]:


import warnings
warnings.filterwarnings('ignore')


# In[58]:


import seaborn as sns
sns.countplot(my_data['Outcome'],label="count")


# In[59]:


#correlation of the data
corr = my_data.corr()
corr


# In[60]:


sns.heatmap(corr,annot=True)


# In[61]:


#Blood pressure : By observing the data we can see that there are 0 values for blood pressure.
# And it is evident that the readings of the data set seems wrong because a living person 
# cannot have diastolic blood pressure of zero. 
print("Total: ",my_data[my_data.BloodPressure == 0].shape[0])
print(my_data[my_data.BloodPressure == 0].groupby('Outcome')['Age'].count())


# In[62]:


#Insulin : In a rare situation a person can have zero insulin
print("Total: ",my_data[my_data.Insulin == 0].shape[0])
print(my_data[my_data.Insulin == 0].groupby('Outcome')['Age'].count())


# In[63]:


# Skin Fold Thickness : For normal people skin fold thickness can’t be less than 10 mm better yet zero.
print("Total: ",my_data[my_data.SkinThickness == 0].shape[0])
print(my_data[my_data.SkinThickness == 0].groupby('Outcome')['Age'].count())


# In[64]:


#BMI : Should not be 0 or close to zero unless the person is really underweight which could be life threatening.
print("Total: ",my_data[my_data.BMI == 0].shape[0])
print(my_data[my_data.BMI == 0].groupby('Outcome')['Age'].count())


# In[65]:


# Plasma glucose levels : Even after fasting glucose level would not be as low as zero.
print("Total: ",my_data[my_data.Glucose == 0].shape[0])
print(my_data[my_data.Glucose == 0].groupby('Outcome')['Age'].count())


# HANDLING INVALID DATA VALUES:

# In[66]:


#remove the rows which the “BloodPressure”, “BMI” and “Glucose” are zero.

my_data=my_data[(my_data.BloodPressure !=0) & (my_data.BMI !=0) & (my_data.Glucose !=0)]
print(my_data.shape)


# In[67]:


from matplotlib import pyplot
import matplotlib.pyplot as plt


# HISTOGRAM:

# In[68]:


fig, axes =plt.subplots(4,2,figsize=(8,8))

sns.histplot(data=my_data["Pregnancies"],kde=True,ax=axes[0,0],color='violet').set(title='Pregnancies Histogram')
sns.histplot(data=my_data["Glucose"],kde=True,ax=axes[0,1],color='indigo').set(title='Glucose Histogram')
sns.histplot(data=my_data["BloodPressure"],kde=True,ax=axes[1,0],color='blue').set(title='BloodPressure Histogram')
sns.histplot(data=my_data["SkinThickness"],kde=True,ax=axes[1,1],color='green').set(title='SkinThickness Histogram')
sns.histplot(data=my_data["Insulin"],kde=True,ax=axes[2,0],color='yellow').set(title='Insulin Histogram')
sns.histplot(data=my_data["BMI"],kde=True,ax=axes[2,1],color='orange').set(title='BMI Histogram')
sns.histplot(data=my_data["DiabetesPedigreeFunction"],kde=True,ax=axes[3,0],color='red').set(title='DiabetesPedigreeFunction Histogram')
sns.histplot(data=my_data["Age"],kde=True,ax=axes[3,1],color='black').set(title='Age Histogram')
plt.tight_layout()
plt.show()


# BOXPLOT:

# In[69]:


fig, axes=plt.subplots(4,2,figsize=(10,10))

sns.boxplot(x=my_data['Pregnancies'], ax=axes[0,0]).set(title='Boxplot for Pregnancies')
sns.boxplot(x=my_data['Glucose'],ax=axes[0,1]).set(title='Boxplot For Glucose')
sns.boxplot(x=my_data['BloodPressure'],ax=axes[1,0]).set(title='Boxplot For Bloodpressure')
sns.boxplot(x=my_data['SkinThickness'],ax=axes[1,1]).set(title='Boxplot of SkinThickness')
sns.boxplot(x=my_data['Insulin'],ax=axes[2,0]).set(title='Boxplot for Insulin')
sns.boxplot(x=my_data['BMI'],ax=axes[2,1]).set(title='Boxplot for BMI')
sns.boxplot(x=my_data['DiabetesPedigreeFunction'],ax=axes[3,0]).set(title='Boxplot for DiabetesPedigreeFunction')
sns.boxplot(x=my_data['Age'],ax=axes[3,1]).set(title='Boxplot for Age')
plt.tight_layout()
plt.show()


# DENSITY PLOT

# In[70]:


my_data.plot(kind='density',subplots=True,layout=(3,3),sharex=False,figsize=(15,15))
pyplot.show()


# SCATTER_MATRIX:

# In[71]:


import pandas
from pandas.plotting import scatter_matrix

dataCorr = my_data.corr()
pandas.plotting.scatter_matrix(dataCorr,figsize=(15,15))
pyplot.show()


# 
