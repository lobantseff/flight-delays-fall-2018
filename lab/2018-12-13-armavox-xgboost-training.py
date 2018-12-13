
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


# ### Init required  modules

# __Notebook environment__

# In[18]:


# pip install watermark
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -r -b -g -p numpy,pandas,sklearn,matplotlib,statsmodels,xgboost,catboost')


# In[96]:


import numpy as np
import pandas as pd
import itertools
import statsmodels.stats.weightstats as wsts
import scipy.stats as stats
from scipy.sparse import hstack, csr_matrix, issparse

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['patch.force_edgecolor'] = True

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                     cross_val_score, GridSearchCV)
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


# ## Data preparation

# ### IMPORT DATA

# In[14]:


train = pd.read_csv('../data/train.csv', index_col='idx')
train.head(3)


# In[15]:


test = pd.read_csv('../data/test.csv', index_col='idx')
test.head(3)


# In[17]:


target = pd.read_csv('../data/target.csv', index_col='idx')
target.head(3)


# __Origin_Dest interaction: ``Route`` feature__

# In[38]:


train['Route'] = train['Origin'] + '_' + train['Dest']
test['Route'] = test['Origin'] + '_' + test['Dest']


# ### FEATURES CONVERSION

# In[68]:


train.head(1)


# In[44]:


ohe = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')


# In[45]:


X_month_train = ohe.fit_transform(train.Month.values.reshape(-1, 1))
X_month_test = ohe.transform(test.Month.values.reshape(-1, 1))


# In[46]:


X_dom_train = ohe.fit_transform(train.DayofMonth.values.reshape(-1, 1))
X_dom_test = ohe.transform(test.DayofMonth.values.reshape(-1, 1))


# In[47]:


X_dow_train = ohe.fit_transform(train.DayOfWeek.values.reshape(-1, 1))
X_dow_test = ohe.transform(test.DayOfWeek.values.reshape(-1, 1))


# In[62]:


X_hour_train = ohe.fit_transform(train.Hour.values.reshape(-1, 1))
X_hour_test = ohe.transform(test.Hour.values.reshape(-1, 1))


# In[63]:


X_minute_train = ohe.fit_transform(train.Minute.values.reshape(-1, 1))
X_minute_test = ohe.transform(test.Minute.values.reshape(-1, 1))


# In[64]:


X_isweekend_train = train.IsWeekend.values.reshape(-1, 1)
X_isweekend_test = test.IsWeekend.values.reshape(-1, 1)


# In[48]:


X_carrier_train = ohe.fit_transform(train.UniqueCarrier.values.reshape(-1, 1))
X_carrier_test = ohe.transform(test.UniqueCarrier.values.reshape(-1, 1))


# In[49]:


X_origin_train = ohe.fit_transform(train.Origin.values.reshape(-1, 1))
X_origin_test = ohe.transform(test.Origin.values.reshape(-1, 1))


# In[50]:


X_dest_train = ohe.fit_transform(train.Dest.values.reshape(-1, 1))
X_dest_test = ohe.transform(test.Dest.values.reshape(-1, 1))


# In[51]:


X_route_train = ohe.fit_transform(train.Route.values.reshape(-1, 1))
X_route_test = ohe.fit_transform(test.Route.values.reshape(-1, 1))


# ### SELECT FEATURES

# In[106]:


def simple_xgb_cv(X, y, n_estimators=27, max_depth=5, seed=42,
                   train_size=0.7):
    """Get ROC-AUC score for simple XGBoost classifier
    """
    skf = StratifiedKFold(n_splits=5, random_state=seed)
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=train_size, random_state=seed)

    xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                        random_state=seed, n_jobs=-1).fit(X_train, y_train)
    
    roc_auc = roc_auc_score(y_valid, xgb.predict_proba(X_valid)[:, 1],
                            average='samples')
    
    cv_score = cross_val_score(xgb, X_train, y_train, scoring='roc_auc', 
                               cv=skf, n_jobs=-1)
    return round(roc_auc, 5), round(cv_score.mean(), 5)


# In[107]:


get_ipython().run_cell_magic('time', '', "features = {'X_month_train': X_month_train, \n            'X_dom_train': X_dom_train, \n            'X_dow_train': X_dow_train, \n            'X_hour_train': X_hour_train, \n            'X_minute_train': X_minute_train,\n            'X_isweekend_train': X_isweekend_train,\n            'X_carrier_train': X_carrier_train,\n            'X_origin_train': X_origin_train,\n            'X_dest_train': X_dest_train,\n            'X_route_train': X_route_train}\n\ndef select_best_feature_comb(features_dict):\n    from scipy.sparse import hstack, csr_matrix, isspmatrix_csr\n    \n    results_dict = {}\n    for l in range(1, len(features)+1):\n\n        list_of_keys = [list(comb) for comb in \n                        itertools.combinations(features.keys(), l)]\n        list_of_combs = [list(comb) for comb in \n                         itertools.combinations(features.values(), l)]\n\n        for keys, comb in zip(list_of_keys, list_of_combs):\n            print('Train on:', keys)\n            \n            is_csr = [isspmatrix_csr(x) for x in comb]\n            if any(is_csr):\n                X_con = hstack(comb, format='csr')\n            else:\n                comb[0] = csr_matrix(comb[0])\n                X_con = hstack(comb, format='csr')\n                \n            result = simple_xgb_cv(X_con, y)\n            results_dict[', '.join(keys)] = result\n            print(f'CV: {result[1]}, OOF: {result[0]}', '\\n')\n            \n    return results_dict\n\nselect_best_feature_comb(features)")


# ### CONCATENATE DATA

# In[60]:


y = target.dep_delayed_15min.values
print("Classes in dataset:", np.unique(y)) 
print('Size:', y.shape)
print("positive objects:", y.sum())


# In[ ]:


X_month_train', 'X_dom_train', 'X_dow_train', 'X_hour_train', 'X_carrier_train


# In[ ]:


X = hstack([X_month_train, X_dom_train,
            X_dow_train, X_] , format='csr')

X_test = hstack([X_tfidf_test, X_hour_test, 
                 X_dow_test, X_daytime_test, 
                 X_timespan_test, X_unique_test,
                 X_intop10_test, X_socnet_test], format='csr')

X.shape, X_test.shape, y.shape


# ## XGBoost TRAINING

# ### Simple XGBoost
