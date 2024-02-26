#!/usr/bin/env python
# coding: utf-8

# # DIABETES DATASET

# In[6]:


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

# In[7]:


my_data=pd.read_csv('diabetes.csv')


# In[8]:


my_data


# In[9]:


my_data.shape


# In[10]:


types=my_data.dtypes
types


# In[39]:


#coloumns
my_data.columns


# In[12]:


#Top 5 rows in dataset
my_data.head()


# MISSING VALUES

# In[13]:


my_data.isnull().sum()


# In[14]:


my_data.info()


# In[40]:


#describing the data
my_data.describe()


# In[16]:


my_data.groupby('Outcome').size()


# In[17]:


#Mean
my_data.groupby('Outcome').mean()


# In[36]:


#Median
my_data.groupby('Outcome').median()


# In[37]:


#Standard deviation
my_data.groupby('Outcome').std()


# In[38]:


#skew calculation
my_data.groupby('Outcome').skew()


# In[21]:


import warnings
warnings.filterwarnings('ignore')


# In[22]:


import seaborn as sns
sns.countplot(my_data['Outcome'],label="count")


# In[23]:


#correlation of the data
corr = my_data.corr()
corr


# In[24]:


sns.heatmap(corr,annot=True)


# In[25]:


#Blood pressure : By observing the data we can see that there are 0 values for blood pressure.
# And it is evident that the readings of the data set seems wrong because a living person 
# cannot have diastolic blood pressure of zero. 
print("Total: ",my_data[my_data.BloodPressure == 0].shape[0])
print(my_data[my_data.BloodPressure == 0].groupby('Outcome')['Age'].count())


# In[26]:


#Insulin : In a rare situation a person can have zero insulin
print("Total: ",my_data[my_data.Insulin == 0].shape[0])
print(my_data[my_data.Insulin == 0].groupby('Outcome')['Age'].count())


# In[27]:


# Skin Fold Thickness : For normal people skin fold thickness can’t be less than 10 mm better yet zero.
print("Total: ",my_data[my_data.SkinThickness == 0].shape[0])
print(my_data[my_data.SkinThickness == 0].groupby('Outcome')['Age'].count())


# In[28]:


#BMI : Should not be 0 or close to zero unless the person is really underweight which could be life threatening.
print("Total: ",my_data[my_data.BMI == 0].shape[0])
print(my_data[my_data.BMI == 0].groupby('Outcome')['Age'].count())


# In[29]:


# Plasma glucose levels : Even after fasting glucose level would not be as low as zero.
print("Total: ",my_data[my_data.Glucose == 0].shape[0])
print(my_data[my_data.Glucose == 0].groupby('Outcome')['Age'].count())


# HANDLING INVALID DATA VALUES:

# In[30]:


#remove the rows which the “BloodPressure”, “BMI” and “Glucose” are zero.

my_data=my_data[(my_data.BloodPressure !=0) & (my_data.BMI !=0) & (my_data.Glucose !=0)]
print(my_data.shape)


# In[31]:


from matplotlib import pyplot
import matplotlib.pyplot as plt


# HISTOGRAM:

# In[32]:


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

# In[33]:


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

# In[34]:


my_data.plot(kind='density',subplots=True,layout=(3,3),sharex=False,figsize=(15,15))
pyplot.show()


# SCATTER_MATRIX:

# In[35]:


import pandas
from pandas.plotting import scatter_matrix

dataCorr = my_data.corr()
pandas.plotting.scatter_matrix(dataCorr,figsize=(15,15))
pyplot.show()


# 
