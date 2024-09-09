#!/usr/bin/env python
# coding: utf-8

# # Introdution

# ## Titanic Dataset - Exploratory Data Analysis (EDA)
# ### Objective:
# To perform the Exploratory Analysis (EDA) on the Titanic dataset to extract meaningful insights about the survival of passengers.

# # Import Libraries:

# ### Importing the Required Libraries

# In[95]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set visualizations to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# set seaborn style
sns.set(style='whitegrid')


# # Loading the Dataset:

# ### Loading the Titanic Dataset
# the Titanic dataset is available in the Seaborn library's built-in dataset. Let's load and explore it

# In[96]:


#Load the dataset using pandas 
dataset = pd.read_csv('Titanic-Dataset.csv')


# Show the first five row of the dataset
dataset.head()


# # Dataset Overview:

# ### Initial Exploration of the Data
# Here we will take a look at the structure of the dataset and understand th columns, datatypes, and any missing values.

# In[97]:


# General information about the dataset
dataset.info()


# In[98]:


# Statistical summary of the dataset
dataset.describe()


# In[99]:


# Checking for missing values
dataset.isnull().sum()


# # Data Cleaning:

# ### Data Cleaning
# we will handle missing values in the dataset by imputing or dropping them as appropriate

# In[100]:


# Fill missing values in 'Age' with the median
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common port
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' columnas it has many missing values
dataset.drop(columns=['Cabin'], inplace=True)

# Check after Cleaning
dataset.isnull().sum()


# In[101]:


# Create a new column 'Is_Child'
dataset['Is_Child'] = np.where(dataset['Age'] < 18, 1, 0)


# # Univariate Analysis:

# ### Univariate Analysis
# We wil analyze individual columns to understand the variables like age, fare, and survival.

# In[102]:


# Age distribution
sns.histplot(dataset['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()


# In[103]:


# Survival Countplot
sns.countplot(x='Survived', data=dataset)
plt.title('Survival count')
plt.show()


# In[104]:


# Fare distribution
sns.histplot(dataset['Fare'], bins=30, kde=True)
plt.title('Fare distribution')
plt.show()


# # Bivariate Analysis:

# ### Bivariate Analysis
# we will explore relationships between two variables , such as servival rate by gender,cchidern/adults and passenger class.

# In[105]:


# Survival rate by gander
sns.barplot(x='Sex', y='Survived', data=dataset)
plt.title('Survival Rate by Gender')
plt.show()


# In[106]:


# Bar plot showing survival rate by whether passenger is a child or not
sns.barplot(x='Is_Child', y='Survived', data=dataset)
plt.title('Survival Rate for Children vs Adults')
plt.xticks([0, 1], ['Adult', 'Child'])
plt.show()


# In[107]:


# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=dataset)
plt.title('Survival Rate by Passenger class')
plt.show()


# # Multivariate Analysis:

# ### Multivariate Analysis
# Multivariate analysis involves exploring more complex relatonships between multiple variables.Here we will analyze survival by class and gender, as well as the relationship between fare and age.

# In[108]:


# Survival rate by class and gender
sns.catplot(x='Pclass', hue='Sex', col='Survived', data=dataset , kind='count')
plt.suptitle('Survival by class and gender', y=1.05)
plt.show()


# In[109]:


# Scatter plot for age vs fare with survival
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=dataset)
plt.title('Fare vs Age with Survival Status')
plt.show()


# # Correlation matrix

# # 8. Correlation Matrix
# A heatmap of the correlation between numerical variables can help us understand the strength of relationships between features.

# In[110]:


# Select only numeric columns from the dataset
numeric_dataset = dataset.select_dtypes(include=['float64', 'int64'])

# Correlation matrix for numerical columns
corr_matrix = numeric_dataset.corr()


# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # Conclusion

# ###  Conclusion
# In this project, we observed:
# - **Gender**: Females had a much higher survival rate compared to males.
# - **Class**: First-class passengers had the highest survival rate.
# - **Age**: Younger passengers, especially children, had better chances of survival.
# - **Fare**: Higher fares are associated with higher survival rates.
# 
# This EDA provided critical insights into the Titanic dataset, which could be further used for predictive modeling.

# In[ ]:




