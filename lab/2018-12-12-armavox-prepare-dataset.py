
# coding: utf-8

# # [Flight delays](https://www.kaggle.com/c/flight-delays-spring-2018/overview)
# Kaggle InClass competiton by [mlcourse.ai](mlcourse.ai). The task is to predict whether a flight will be delayed for more than 15 minutes

# ## Initialization

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


# ## Data import and initial research

# ### IMPORT

# In[2]:


get_ipython().system('ls ../raw_competition_data/')


# In[3]:


train = pd.read_csv('../raw_competition_data/flight_delays_train.csv.zip', 
                    engine='c')
test = pd.read_csv('../raw_competition_data/flight_delays_test.csv.zip', 
                   engine='c')


# In[4]:


_addr = '../raw_competition_data/sample_submission.csv.zip'
sample_submission = pd.read_csv(_addr, engine='c')
sample_submission.head(1)


# Features:
# 
# - **Month, DayofMonth, DayOfWeek**
# - **DepTime** – departure time
# - **UniqueCarrier** – code of a company-career
# - **Origin** – flight origin
# - **Dest** – flight destination
# - **Distance** - distance between Origin and Dest airports
# - **dep_delayed_15min** – target

# In[5]:


train['dep_delayed_15min'] = train['dep_delayed_15min'].map({'N': 0, 'Y': 1})
train.head()


# In[6]:


# Table to picture the origin_dest delay dependencies. (It will happen in 
# the "Research" section)
origin_dest_table = train.pivot_table(index='Origin', columns='Dest', 
                                      values='dep_delayed_15min', 
                                      aggfunc='sum', fill_value=0)


# Some feature engineering

# In[7]:


num_cols = ['Distance']
datetime_cols = ['Month','DayofMonth','DayOfWeek']
cat_cols = ['UniqueCarrier', 'Origin', 'Dest']
train[num_cols] = train[num_cols].astype('float')

# Remove `c-` from the date cols
train[datetime_cols] = train[datetime_cols].applymap(lambda x: x[2:])
train[datetime_cols] = train[datetime_cols].astype('int')

# Extract Hour and Minute of departure from the `DepTime`
train['Hour'] = train['DepTime'].apply(lambda x: x // 100)
train['Hour'] = train['Hour'].astype('int')
train['Minute'] = train['DepTime'].apply(lambda x: x % 100)
train['Minute'] = train['Minute'].astype('int')
train.drop('DepTime', axis=1, inplace=True)

# During visual analysis has been founded that six objects has DepTime hour 
# equal to 25. I decided to drop such objects
train = train[train.Hour != 25]

train['IsWeekend'] = train['DayOfWeek'].isin([6,7]).astype('int')


# In[8]:


train[train.Hour ==25]


# In[9]:


train.head()


# In[10]:


target = train[['dep_delayed_15min']]
train.drop('dep_delayed_15min', axis=1, inplace=True)


# In[11]:


test.head(3)


# In[12]:


test.info()


# In[13]:


test[num_cols] = test[num_cols].astype('float')
test[datetime_cols] = test[datetime_cols].applymap(lambda x: x[2:])
test[datetime_cols] = test[datetime_cols].astype('int')

test['Hour'] = test['DepTime'].apply(lambda x: x // 100)
test['Hour'] = test['Hour'].astype('int')
test['Minute'] = test['DepTime'].apply(lambda x: x % 100)
test['Minute'] = test['Minute'].astype('int')
test.drop('DepTime', axis=1, inplace=True)

test['IsWeekend'] = test['DayOfWeek'].isin([6,7]).astype('int')


# In[14]:


test.head(3)


# In[15]:


test.info()


# In[16]:


datetime_cols.append('Hour')
datetime_cols.append('Minute')


# ### RESEARCH

# #### Class balance

# In[17]:


plt.figure(figsize=(12,2))
sns.countplot(y=target.dep_delayed_15min);


# Our dataset is unbalanced, hense we should to take some action to overhead this problem in the case of training decision trees or related models.

# #### Datetime features distributions

# In[18]:


fig, axes = plt.subplots(5, 4, figsize=(27, 3*5))

for ax, col in zip(axes, datetime_cols):
    _plot_data = train
    s = sns.countplot(data=train[target.dep_delayed_15min==0], 
                  x=col, ax=ax[0], color='silver')
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    s = sns.countplot(data=train[target.dep_delayed_15min==1], 
                  x=col, ax=ax[1], color='tomato')
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    s = sns.countplot(data=train, x=col, ax=ax[2], color='steelblue')
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    s = sns.countplot(data=test, x=col, ax=ax[3], color='orange')
    s.set_xticklabels(s.get_xticklabels(), rotation=90)

    
# Column row names
cols = ['Target 0 distribution', 
        'Target 1 distribution',
        'Train set feature distribution',
        'Test set feature distribution']
rows = datetime_cols
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, labelpad=12, size='large')
    
fig.tight_layout()


# We see, that in the train and in  the test sets data is distrubuted in a similar way, hence we can use this features o predict target on the test data. In addition, We can notice, that target data in the every feature has a bit different distribution, what is a good news to us – it means, that target data can be distinguished in some way in all data by these features.

# In[19]:


plt.figure(figsize=(16,4))
train.groupby('Minute').size().plot(kind='bar', color='steelblue')
plt.title('Number of departures by minute of every hour');


# Here is some

# #### Unique carrier distribution between train and dataset

# In[20]:


train_carrier = train[['UniqueCarrier']].copy()
train_carrier['set'] = 'train'
train_carrier['target'] = target['dep_delayed_15min'].values
test_carrier = test[['UniqueCarrier']].copy()
test_carrier['set'] = 'test'
unique_carrier = pd.concat([train_carrier[['UniqueCarrier', 'set']], 
                            test_carrier])

plt.figure(figsize=(12,3))
s = sns.countplot(data=unique_carrier, x='UniqueCarrier', hue='set')
s.legend(title='dataset', loc=1);


# #### Target distribution on UniqueCarrier feature

# In[21]:


plt.figure(figsize=(12,3))
sns.countplot(data=train_carrier, x='UniqueCarrier', hue='target');


# #### Origin -> Destination delay
# In this figure, we cannot specify exact flights that are usually delayed, but we see that delays are not uniformly distributed, and, moreover, there are several departure and destination airports with constant flight delays.

# In[22]:


plt.figure(figsize=(20,16))
sns.heatmap(origin_dest_table, square=True)
plt.title('Number of delays for origin-destination pairs');


# ### EXPORT TRAIN/TEST DATASET AND TARGET

# In[23]:


train.head()


# In[24]:


test.head()


# In[25]:


target.head()


# In[26]:


train.shape, test.shape, target.shape


# In[27]:


train.to_csv('../data/train.csv', index_label='idx')
test.to_csv('../data/test.csv', index_label='idx')
target.to_csv('../data/target.csv', index_label='idx')

