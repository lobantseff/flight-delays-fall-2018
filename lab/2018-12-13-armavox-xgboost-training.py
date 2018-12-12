
# coding: utf-8

# # [Flight delays](https://www.kaggle.com/c/flight-delays-spring-2018/overview)
# Kaggle InClass competiton by [mlcourse.ai](mlcourse.ai). The task is to predict whether a flight will be delayed for more than 15 minutes

# ## Initialization

# ### Perform dataset preparing

# In[7]:


# pip install nbformat
# execute `output.show()` to show output figures and text (if they are)
%%capture output  
get_ipython().run_line_magic('run', './2018-12-12-armavox-prepare-dataset.ipynb')


# In[1]:


import numpy as np
import pandas as pd
import statsmodels.stats.weightstats as wsts
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['patch.force_edgecolor'] = True

# pip install watermark
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -r -b -g -p numpy,pandas,sklearn,matplotlib,statsmodels,xgboost,catboost')


# ## Training

# ### IMPORT DATA

# In[ ]:


train = pd.read_csv('../')

