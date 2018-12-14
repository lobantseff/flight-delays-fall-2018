
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

# In[1]:


# pip install watermark
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -r -b -g -p numpy,pandas,sklearn,matplotlib,statsmodels,xgboost,catboost')


# In[2]:


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

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


# ## Data preparation

# ### IMPORT DATA

# In[3]:


train = pd.read_csv('../data/train.csv', index_col='idx')
train.head(3)


# In[4]:


test = pd.read_csv('../data/test.csv', index_col='idx')
test.head(3)


# In[5]:


target = pd.read_csv('../data/target.csv', index_col='idx')
target.head(3)


# __Origin_Dest interaction: ``Route`` feature__

# In[6]:


train['Route'] = train['Origin'] + '_' + train['Dest']
test['Route'] = test['Origin'] + '_' + test['Dest']


# ### FEATURES CONVERSION

# In[7]:


train.head(1)


# In[8]:


ohe = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')


# In[9]:


X_month_train = ohe.fit_transform(train.Month.values.reshape(-1, 1))
X_month_test = ohe.transform(test.Month.values.reshape(-1, 1))


# In[10]:


X_dom_train = ohe.fit_transform(train.DayofMonth.values.reshape(-1, 1))
X_dom_test = ohe.transform(test.DayofMonth.values.reshape(-1, 1))


# In[11]:


X_dow_train = ohe.fit_transform(train.DayOfWeek.values.reshape(-1, 1))
X_dow_test = ohe.transform(test.DayOfWeek.values.reshape(-1, 1))


# In[12]:


X_hour_train = ohe.fit_transform(train.Hour.values.reshape(-1, 1))
X_hour_test = ohe.transform(test.Hour.values.reshape(-1, 1))


# In[13]:


X_minute_train = ohe.fit_transform(train.Minute.values.reshape(-1, 1))
X_minute_test = ohe.transform(test.Minute.values.reshape(-1, 1))


# In[14]:


X_isweekend_train = train.IsWeekend.values.reshape(-1, 1)
X_isweekend_test = test.IsWeekend.values.reshape(-1, 1)


# In[15]:


X_carrier_train = ohe.fit_transform(train.UniqueCarrier.values.reshape(-1, 1))
X_carrier_test = ohe.transform(test.UniqueCarrier.values.reshape(-1, 1))


# In[16]:


X_origin_train = ohe.fit_transform(train.Origin.values.reshape(-1, 1))
X_origin_test = ohe.transform(test.Origin.values.reshape(-1, 1))


# In[17]:


X_dest_train = ohe.fit_transform(train.Dest.values.reshape(-1, 1))
X_dest_test = ohe.transform(test.Dest.values.reshape(-1, 1))


# In[18]:


X_route_train = ohe.fit_transform(train.Route.values.reshape(-1, 1))
X_route_test = ohe.fit_transform(test.Route.values.reshape(-1, 1))


# ### SELECT FEATURES

# In[19]:


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


# In[ ]:


get_ipython().run_cell_magic('time', '', "features = {'X_month_train': X_month_train, \n            'X_dom_train': X_dom_train, \n            'X_dow_train': X_dow_train, \n            'X_hour_train': X_hour_train, \n            'X_minute_train': X_minute_train,\n            'X_isweekend_train': X_isweekend_train,\n            'X_carrier_train': X_carrier_train,\n            'X_origin_train': X_origin_train,\n            'X_dest_train': X_dest_train,\n            'X_route_train': X_route_train}\n\ndef select_best_feature_comb(features_dict):\n    from scipy.sparse import hstack, csr_matrix, isspmatrix_csr\n    \n    results_dict = {}\n    for l in range(1, len(features)+1):\n\n        list_of_keys = [list(comb) for comb in \n                        itertools.combinations(features.keys(), l)]\n        list_of_combs = [list(comb) for comb in \n                         itertools.combinations(features.values(), l)]\n\n        for keys, comb in zip(list_of_keys, list_of_combs):\n            print('Train on:', keys)\n            \n            is_csr = [isspmatrix_csr(x) for x in comb]\n            if any(is_csr):\n                X_con = hstack(comb, format='csr')\n            else:\n                comb[0] = csr_matrix(comb[0])\n                X_con = hstack(comb, format='csr')\n                \n            result = simple_xgb_cv(X_con, y)\n            results_dict[', '.join(keys)] = result\n            print(f'CV: {result[1]}, OOF: {result[0]}', '\\n')\n            \n    return results_dict\n\nselect_best_feature_comb(features)")


# ### CONCATENATE DATA

# In[13]:


y = target.dep_delayed_15min.values
print("Classes in dataset:", np.unique(y)) 
print('Size:', y.shape)
print("positive objects:", y.sum())
balance_coef = np.sum(y==0) /  np.sum(y==1)


# In[21]:


X = hstack([csr_matrix(X_month_train), 
            X_dom_train, 
            X_dow_train, 
            X_hour_train, 
            X_minute_train,
            X_isweekend_train,
            X_carrier_train,
            X_origin_train,
            X_dest_train,
            X_route_train] , format='csr')

X_test = hstack([csr_matrix(X_month_test), 
            X_dom_test, 
            X_dow_test, 
            X_hour_test, 
            X_minute_test,
            X_isweekend_test,
            X_carrier_test,
            X_origin_test,
            X_dest_test,
            X_route_test], format='csr')

X.shape, X_test.shape, y.shape


# ## CatBoost TRAINING

# ### Simple Random forest

# In[22]:


X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.7, random_state=42)


# In[29]:


get_ipython().run_cell_magic('time', '', 'rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score=True)\nrf.fit(X_train, y_train)')


# In[30]:


rf.oob_score_


# In[31]:


roc_auc_score(y_valid, rf.predict_proba(X_valid)[:,1])


# ### Simple CatBoost

# In[7]:


from catboost import CatBoostClassifier, Pool
from catboost import cv


# In[8]:


cat_features_idx = np.where((train.dtypes == 'object') | (train.dtypes == 'int64'))[0].tolist()
cat_features_idx


# In[9]:


X_train, X_valid, y_train, y_valid = train_test_split(
        train, target, train_size=0.7, random_state=42)


# In[10]:


cat = CatBoostClassifier(random_state=42, thread_count=72)


# In[49]:


get_ipython().run_line_magic('time', 'cat.fit(X_train, y_train, cat_features=cat_features_idx)')


# In[50]:


roc_auc_score(y_valid, cat.predict_proba(X_valid)[:,1])


# In[51]:


get_ipython().run_cell_magic('time', '', "pool = Pool(train, target, cat_features=cat_features_idx)\nparams = {'loss_function': 'Logloss', \n          'verbose': False,\n          'thread_count': 72,\n          'custom_metric': 'AUC'}\ncat_cv = cv(pool, params, fold_count=5, stratified=True, seed=42, num_boost_round=1500,\n            early_stopping_rounds=10, plot=True)")


# ### Catboost search

# In[49]:


X_train, X_valid, y_train, y_valid = train_test_split(
        train, target, train_size=0.95, random_state=42)
train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
eval_pool = Pool(X_valid, y_valid, cat_features=cat_features_idx)


# In[ ]:


cat_search_params = {
    'depth' : np.arange(1,17),
    'l2_leaf_reg' : np.linspace(1, 5, 9),
#     'rsm': np.linspace(0.5, 1, 3),
    }


# In[57]:


cat_params = {
    'iterations' : 1200,
    'learning_rate' : 0.05,
    'depth' : 10,
    'l2_leaf_reg' : 10,
    'loss_function' : 'Logloss',
    'border_count' : 254,
    'od_type': 'IncToDec',
    'od_pval' : 1e-5,
#     'early_stopping_rounds' : 10,
    'random_seed' : 42,
    'use_best_model' : True,
    'verbose': False,
    'scale_pos_weight' : balance_coef,
    'random_strength' : 1,
    'custom_metric' : 'AUC',
    'eval_metric' : 'AUC',
    'boosting_type' : 'Ordered',
#     'bagging_temperature' : 3,
#     'subsample' : 0.66,
    'cat_features' : cat_features_idx ,
    'save_snapshot': True,
    'snapshot_file': 'depth_16'
}

cat = CatBoostClassifier(**cat_params)


# In[ ]:


cat.fit(train_pool, use_best_model=True, eval_set=eval_pool, plot=True,
       save_snapshot=True, snapshot_file='finalfinal', snapshot_interval=60)


# In[ ]:


roc_auc_score(y_valid, 
              cat.predict_proba(X_valid)[:, 1])


# In[ ]:


cat.best_iteration_


# ## SUBMISSION

# ### Last check

# In[ ]:


final_estimator = cat

X_train, X_valid, y_train, y_valid = train_test_split(train, target, 
                                                      train_set=0.9)


final_estimator.fit(X_train, y_train)
_roc_auc = roc_auc_score(y_valid, 
                         final_estimator.predict_proba(X_valid)[:, 1])

print(f'ROC-AUC: {_roc_auc:.5f}')
# ### Train on the full dataset

# In[56]:


get_ipython().run_cell_magic('time', '', 'pool = Pool(train, target, cat_features=cat_features_idx)\nfinal_estimator.fit(train_pool, use_best_model=True, eval_set=eval_pool,\n                    plot=True, save_snapshot=True,\n                    snapshot_interval=60)\nfinal_pred = final_estimator.predict_proba(X_test)[: 1]')


# ### Write submission file

# In[63]:


# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file, 
                             target='dep_delayed_15min', index_label="id"):
    
    predicted_df = pd.DataFrame(
        predicted_labels,
        index = np.arange(0, predicted_labels.shape[0]),
        columns=[target])
    
    predicted_df.to_csv(out_file, index_label=index_label)


# In[64]:


from datetime import datetime as dt
import subprocess
now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
label = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")

### WRITE SUBMISSION
write_to_submission_file(final_pred, f'../submissions/catboost_submission_at_{now}__githash_{label}.csv')

pd.Series(final_pred, 
          name='dep_delayed_15min').to_csv('xgb_2feat.csv', 
                                           index_label='id', header=True)


# In[58]:


pd.Series(final_pred, 
          name='dep_delayed_15min').to_csv('xgb_2feat.csv', 
                                           index_label='id', header=True)


# In[2]:


pd.read_csv('../submissions/catboost_submission_at_2018-12-13_00-13-17__githash_2a09165.csv')

