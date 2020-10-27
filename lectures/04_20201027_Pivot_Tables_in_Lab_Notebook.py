#!/usr/bin/env python
# coding: utf-8

# # Lecture #4: Pivot Tables in a Lab Notebook
# 
# _October 27 2020_

# ## Why notebooks?
# 
# The most important reasons are listed here: http://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261
# 
# Open this file in [Jupyter Notebook/Lab](https://jupyter.org/), [Google Colab](https://colab.research.google.com/), or similar.

# ## Excel tutorial example
# 
# See previous lecture notes for the context. Let's the import the data set again, directly from the Excel file to [pandas](https://pandas.pydata.org/):

# In[1]:


import pandas as pd


# In[2]:


xlsx = pd.read_excel('files/excel2016_intropivottables_practice.xlsx')


# Review the data frame:

# In[3]:


xlsx.info()


# In[4]:


xlsx.head()


# Example analysis:

# In[5]:


xlsx.pivot_table(
    index=['Salesperson'],
    columns=['Month'],
    values=['Order Amount'],
    aggfunc=sum
)


# Let's fix the inappropropriate sorting of months:

# In[6]:


xlsx['Month'] = xlsx['Month'].apply(lambda m: '{}_{}'.format(1 if m == 'January' else 2 if m == 'February' else 3, m))
xlsx.head()


# Note: Better way would be to treat it is properly as datetime, we could use the [to_datetime()](http://pandas.pydata.org/pandas-docs/version/0.20/generated/pandas.to_datetime.html) function.

# In[7]:


xlsx.pivot_table(
    index=['Salesperson'],
    columns=['Month'],
    values=['Order Amount'],
    aggfunc=sum
)


# Here, we can see performance of agents across the time dimension (grouped on monthly level). 
# 
# **What about plotting this analysis?**

# In[8]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


xlsx.pivot_table(
    index=['Month'],
    columns=['Salesperson'],
    values=['Order Amount'],
    aggfunc=sum
)['Order Amount'].plot(style='.-', figsize=(8, 5));


# ## Car mpg data set exploaration example
# 
# Downloaded from https://archive.ics.uci.edu/ml/datasets/Auto+MPG.
# 
# File preview (in bash):

# In[10]:


get_ipython().system('head files/auto-mpg.data.txt')


# Import the fixed width format:

# In[11]:


mpg = pd.read_fwf('files/auto-mpg.data.txt', 
                  na_values='?',  # see documentation
                  header=0, 
                  names=['mpg', 'cylinders', 'displacement', 'horsepower',
                         'weight', 'acceleration', 'model year', 'origin', 'car name'])
                  
mpg.info()


# Mind the NA (NULL) values in the `horsepower` variable.

# In[12]:


mpg.head()


# Basic description:

# In[13]:


mpg.describe()


# Useful for quick frequency calculation:

# In[14]:


mpg['origin'].value_counts()


# In[15]:


mpg['cylinders'].value_counts().sort_index()


# ### Displacement (bins) vs horse power?

# In[16]:


import numpy as np


# Prepare the bins, as in the Excel groping:

# In[17]:


bin_size = 50
disp_min = mpg['displacement'].min()
disp_max = mpg['displacement'].max()
bins = [disp_min + bin_size * s for s in np.arange(np.ceil((disp_max - disp_min) / bin_size))]


# In[18]:


bins


# **Mean horse power by displacement bins**:

# In[19]:


hp = mpg.groupby(np.digitize(mpg['displacement'], bins))['horsepower'].mean()
hp


# In[20]:


hp.plot(style='.');


# **Linear regression**:

# In[21]:


import statsmodels.api as sm


# In[22]:


y = hp.values
y


# In[23]:


X = hp.index.values
X


# In[24]:


X = sm.add_constant(X)
X


# In[25]:


model = sm.OLS(y, X)
results = model.fit()


# In[26]:


results.params


# Fit plot:

# In[27]:


fig = sm.graphics.plot_fit(results, 'x1')
fig.tight_layout(pad=1.0)


# In[28]:


print(results.summary())


# See more on [OLS](http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html).

# ## Assignment
# 
# 1. Do similar analysis on your own data set
# 2. Use [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) to inspect the data set
