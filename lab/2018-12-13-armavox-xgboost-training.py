
# coding: utf-8

# # [Flight delays](https://www.kaggle.com/c/flight-delays-spring-2018/overview)
# Kaggle InClass competiton by [mlcourse.ai](mlcourse.ai). The task is to predict whether a flight will be delayed for more than 15 minutes

# ## Initialization

# ### Perform dataset preparing

# In[2]:


get_ipython().run_cell_magic('capture', 'output  ', '# pip install nbformat\n# execute `output.show()` to show output figures and text (if they are)\n%run ./2018-12-12-armavox-prepare-dataset.ipynb')


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

from xgboost import XGBClassifier

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

# In[ ]:


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

# In[ ]:


y = target.dep_delayed_15min.values
print("Classes in dataset:", np.unique(y)) 
print('Size:', y.shape)
print("positive objects:", y.sum())
balance_coef = np.sum(y==0) /  np.sum(y==1)


# In[ ]:


# Best feature combination

X = np.hstack([
    X_month_train,
    X_dom_train,
    X_dow_train,
    X_hour_train,
    X_minute_train,
    X_isweekend_train,
    X_carrier_train,
#     X_origin_train
    X_dest_train,
#     X_route_train
])

X_test = np.hstack([
    X_month_test,
    X_dom_test,
    X_dow_test,
    X_hour_test,
    X_minute_test,
    X_isweekend_test,
    X_carrier_test,
#     X_origin_test,
    X_dest_test,
#     X_route_test
])

X.shape, X_test.shape, y.shape


# ## XGBoost TUNING

# ### Simple XGBoost

# In[96]:


X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.9, random_state=42)
skf = StratifiedKFold(n_splits=5, random_state=42)


# In[ ]:


xgb = XGBClassifier(n_estimators=300, max_depth=3, random_state=42, n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('capture', 'In24', '%%time\nxgb.fit(X_train, y_train)\nprint(roc_auc_score(y_valid, xgb.predict_proba(X_valid)[:, 1]))')


# In[ ]:


get_ipython().run_cell_magic('capture', 'In25', "_cv_score = cross_val_score(xgb, X, y, scoring='roc_auc', cv=skf, n_jobs=-1)\n_cv_score.mean(), _cv_score.std()")


# In[29]:


In25.show()


# ### XGB CV

# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# #### Iteration #1. Model complexity

# In[33]:


_xgb_grid_params_iteration1 = {
    'colsample_bytree': np.linspace(0.4, 1, 5),
    'gamma': np.linspace(0.5, 1, 5),
    'max_depth': np.arange(1, 11),
    'min_child_weight': np.arange(1,11),
    'reg_alpha': np.logspace(-2, 2, 8),
    'reg_lambda': np.logspace(-2, 2, 8),
    'subsample': np.linspace(0.5, 1, 8)
}


# In[ ]:


get_ipython().run_cell_magic('capture', 'In28', "%%time\nxgb = XGBClassifier(n_estimators=30, \n                    scale_pos_weight=balance_coef,\n                    random_state=42, n_jobs=-1)\n\nxgb_search1 = RandomizedSearchCV(xgb, _xgb_grid_params_iteration1, \n                                 n_iter=1000, cv=skf, scoring='roc_auc', \n                                 random_state=42, n_jobs=-1, verbose=1)\nxgb_search1.fit(X_train, y_train)")


# In[ ]:


get_ipython().run_cell_magic('capture', 'In29', 'xgbest1 = xgb_search1.best_estimator_\nxgb_best_complexity = xgb_search1.best_params_\nxgb_search1.best_score_, xgb_search1.best_params_')


# In[ ]:


get_ipython().run_cell_magic('capture', 'In30', 'print(f"""ROC-AUC on the validation data: \n{roc_auc_score(y_valid, xgbest1.predict_proba(X_valid)[:, 1]):.5f}""")')


# In[45]:


In29.show()


# In[44]:


In30.show()


# In[49]:


import xgboost as xgb


# #### Iteration #2. Model optimization

# In[50]:


dtrain = xgb.DMatrix(X, y)


# In[81]:


best_params = xgb_best_complexity
best_params['objective'] = 'binary:logistic'
best_params['nthread'] = 16
# best_params['silent'] = 1
best_params['eval_metric'] = 'auc'
best_params['eta'] = 0.05
best_params


# In[75]:


get_ipython().run_cell_magic('time', '', "xgb_cv = xgb.cv(best_params, dtrain, num_boost_round=500, metrics='auc',\n                stratified=True, nfold=5, early_stopping_rounds=50, \n                verbose_eval=True, seed=42)")


# In[77]:


plt.plot(range(xgb_cv.shape[0]), xgb_cv['test-auc-mean'], label='test')
plt.plot(range(xgb_cv.shape[0]), xgb_cv['train-auc-mean'], label='train')
plt.legend();


# In[ ]:


best_num_round = np.argmin(xgb_cv['test-auc-mean'])


# #### hyperopt optimization

# In[112]:


from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[128]:


X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.7, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[150]:


def score(params):
    print("Training with params:")
    print(params)
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid)
    params['max_depth'] = int(params['max_depth'])
    model = xgb.train(params, dtrain, params['num_round'])
    predictions = model.predict(dvalid).reshape(-1, 1)
    score = 1 - roc_auc_score(y_valid, predictions)
    print("\tScore {0}\n\n".format(score))
    return {'loss': score, 'status': STATUS_OK}


# In[151]:


def optimize(trials):
    space = {
             'num_round': 30,
             'learning_rate': 0.05,
             'max_depth': hp.quniform('max_depth', 3, 14, 1),
             'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
             'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma': hp.quniform('gamma', 0.5, 1, 0.01),
             'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.05),
             'eval_metric': 'auc',
             'objective': 'binary:logistic',
             'nthread' : 8,
             'silent' : 1
             }
    
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=10)
    return best


# In[152]:


trials = Trials()
best_params = optimize(trials)
best_params


# In[158]:


best_params['max_depth'] = int(best_params['max_depth'])
best_params['eta']= 0.05
best_params['eval_metric']= 'auc'
best_params['objective']= 'binary:logistic'
best_params['nthread'] = 8
best_params['silent'] = 1
best_params


# In[159]:


get_ipython().run_cell_magic('time', '', "xgb_cv = xgb.cv(best_params, dtrain, num_boost_round=500, metrics='auc',\n                stratified=True, nfold=5, early_stopping_rounds=50, \n                verbose_eval=True, seed=42)")


# In[160]:


plt.plot(range(xgb_cv.shape[0]), xgb_cv['test-auc-mean'], label='test')
plt.plot(range(xgb_cv.shape[0]), xgb_cv['train-auc-mean'], label='train')
plt.legend();


# In[162]:


best_num_round = np.argmax(xgb_cv['test-auc-mean'])
best_num_round


# In[163]:


X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.9, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[164]:


dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid)
dfull = xgb.DMatrix(X, y)
dtest = xgb.DMatrix(X_test)


# In[165]:


get_ipython().run_cell_magic('time', '', 'bestXgb = xgb.train(best_params, dtrain, num_boost_round=best_num_round)\nprint(roc_auc_score(y_valid, bestXgb.predict(dvalid).reshape(-1,1)))')


# In[166]:


fullXGB = xgb.train(best_params, dfull, num_boost_round=best_num_round)
final_pred = fullXGB.predict(dtest)


# In[ ]:


_xgb_grid_params_iteration2 = {
    'n_estimators': np.linspace(100, 1000, 10, dtype='int'),
#     'learning_rate': np.arange(0.005, 0.1, 0.005)
}


# In[ ]:


get_ipython().run_cell_magic('capture', 'In32', "%%time\n\nxgb_search2 = GridSearchCV(xgbest1, _xgb_grid_params_iteration2,\n                                cv=skf, scoring='roc_auc', \n                                n_jobs=-1, verbose=1)\nxgb_search2.fit(X_train, y_train)")


# In[46]:


In32.show()


# In[ ]:


get_ipython().run_cell_magic('capture', 'In33', 'xgbest2 = xgb_search2.best_estimator_\nxgb_best_complexity = xgb_search2.best_params_\nxgb_search2.best_score_, xgb_search2.best_params_')


# In[ ]:


get_ipython().run_cell_magic('capture', 'In34', 'print(f"""ROC-AUC on the validation data: \n{roc_auc_score(y_valid, xgbest2.predict_proba(X_valid)[:, 1]):.5f}""")')


# ## SUBMIT

# ### Last check

# In[ ]:


X.shape, X_test.shape, y.shape


# In[ ]:


final_estimator = xgbest2
final_estimator


# In[ ]:


get_ipython().run_cell_magic('capture', 'In37', "X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_set=0.9, \n                                                      random_state=42)\n\n_cv_score = cross_val_score(final_estimator, X, y, scoring='roc_auc', cv=skf,\n                            n_jobs=-1)\n\nfinal_estimator.fit(X_train, y_train)\n_roc_auc = roc_auc_score(y_valid, final_estimator.predic_proba(X_valid)[:, 1])\n\nprint(f'CV: {_cv_score:.5f} \\n ROC-AUC: {_roc_auc}')")


# ### Train on the full dataset

# In[ ]:


get_ipython().run_cell_magic('capture', 'In38', '%%time\nfinal_estimator.fit(X, y)\nfinal_pred = final_estimator.predict_proba(X_test)[: 1]')


# ### Write submission

# In[169]:


# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file, 
                             target='dep_delayed_15min', index_label="id"):
    
    predicted_df = pd.DataFrame(
        predicted_labels,
        index = np.arange(0, predicted_labels.shape[0]),
        columns=[target])
    
    predicted_df.to_csv(out_file, index_label=index_label)


# In[175]:


get_ipython().system('git describe --always')


# In[174]:


subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")


# In[172]:


from datetime import datetime as dt
import subprocess
now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
label = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")

### WRITE SUBMISSION
# write_to_submission_file(final_pred, f'../submissions/xgb_submission_at_{now}__githash_{label}.csv')

