import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transformers import ColumnExtractor, DFStandardScaler, DFFeatureUnion
from transformers import AddGroupByCount, DummyTransformer, LabelTransformer, Log1pTransformer, ZeroFillTransformer
from transformers import DateFormatter, DateToMonth, DateDiffer, DeleteCol
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

import h2o
from h2o.transforms.preprocessing import H2OScaler
from h2o.transforms.decomposition import H2OPCA
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn2pmml.preprocessing.h2o import H2OFrameCreator

##############################
# Load Data
##############################
# Dir_PATH = ''
# File_PATH = Dir_PATH + 'Fraud_Data.csv'
# df = pd.read_csv(File_PATH)
# File_PATH = Dir_PATH + 'IpAddress_to_Country.csv'
# ipToCountry = pd.read_csv(File_PATH)
Dir_PATH = ''
File_PATH = Dir_PATH + 'df.csv'
df = pd.read_csv(File_PATH)

##############################
# Quick Look
##############################
# print(df.head())
# print(ipToCountry.head())
# print(df.info())
# print(ipToCountry.info())
# print(df.describe())
# print(ipToCountry.describe())
# print(df.isnull().sum())
# print(ipToCountry.isnull().sum())
# # check uniqueness
# duplicateRowsDF = df[df.duplicated(['user_id', 'purchase_time'])]
# print(duplicateRowsDF)
# # no duplicate rows

##############################
# Convert IP to Country
##############################
# for i in range(len(df.index)):
#     ip = df.at[i, 'ip_address']
#     for j in range(len(ipToCountry.index)):
#         if ip >= ipToCountry.at[j, 'lower_bound_ip_address'] and ip <= ipToCountry.at[j, 'upper_bound_ip_address']:
#             df.at[i, 'country'] = ipToCountry.at[j, 'country']

# countries = []
# for i in range(len(df)):
#     ip_address = df.loc[i, 'ip_address']
#     tmp = ipToCountry[(ipToCountry['lower_bound_ip_address'] <= ip_address)
#                       & (ipToCountry['upper_bound_ip_address'] >= ip_address)]
#     print(tmp)
#     if len(tmp) == 1:
#         countries.append(tmp['country'].values[0])
#     else:
#         countries.append('NA')
# df['country'] = countries
# print(df.isnull().sum())
# df.to_csv('df.csv', index=False)

##############################
# Features
##############################
# Group columns by type of preprocessing needed
OUTCOME = 'class'
NEAR_UNIQUE_FEATS = ['user_id', 'device_id']
NUM_FEATS = ['purchase_value', 'age', 'ip_address']
CAT_OHE_FEATS = ['source', 'browser', 'sex']
CAT_LAB_FEATS = ['country']
DATE_FEATS = ['signup_time', 'purchase_time']

##############################
# Data Splitting
##############################
# Set aside 25% as test data
X = df.drop([OUTCOME], axis=1)
Y = df[OUTCOME]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30, stratify=Y)
# print ("train feature shape: ", X_train.shape)
# print ("test feature shape: ", X_test.shape)

##############################
# EDA
##############################
y_train_temp = y_train.reset_index(drop=True)
X_train_temp = X_train.reset_index(drop=True)
train = pd.concat([X_train_temp, y_train_temp], axis=1)
# # time
# train['signup_time'] = pd.to_datetime(train['signup_time'], errors='coerce')
# train['signup_month'] = train['signup_time'].dt.month
# train['signup_weekday'] = train['signup_time'].dt.weekday
# train['signup_hour'] = train['signup_time'].dt.hour
# signup_month = train.groupby(['signup_month']).size()
# signup_month.index = signup_month.index.astype('int', copy=False)
# signup_weekday = train.groupby(['signup_weekday']).size()
# signup_weekday.index = signup_weekday.index.astype('int', copy=False)
# signup_hour = train.groupby(['signup_hour']).size()
# signup_hour.index = signup_hour.index.astype('int', copy=False)
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(100, 100))
# ax[0, 0].plot(signup_month, 'r')
# ax[0, 0].set_title('signup_month')
# ax[1, 0].plot(signup_weekday, 'b')
# ax[1, 0].set_title('signup_weekday')
# ax[0, 1].plot(signup_hour, 'g')
# ax[0, 1].set_title('signup_hour')
# plt.show()
# hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
# fig, ax = plt.subplots(figsize=(15, 7))
# sns.distplot(train[train['class']==0]['signup_month'],
#              label='Not Fraud', ax=ax, hist_kws=hist_kws)
# sns.distplot(train[train['class']==1]['signup_month'],
#              label='Fraud', ax=ax, hist_kws=hist_kws)
# ax.set_xlabel('weekday', fontsize=12)
# ax.set_ylabel('PDF', fontsize=12)
# ax.legend()
# plt.show()
# # add signup_month as new feature
# train['purchase_time'] = pd.to_datetime(train['purchase_time'], errors='coerce')
# train['interval_ins'] = map(lambda x: x.total_seconds(), train['purchase_time'] - train['signup_time'])
# hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
# fig, ax = plt.subplots(figsize=(15, 7))
# sns.distplot(train[train['class']==0]['interval_ins'],
#              label='Not Fraud', ax=ax, hist_kws=hist_kws)
# sns.distplot(train[train['class']==1]['interval_ins'],
#              label='Fraud', ax=ax, hist_kws=hist_kws)
# ax.set_xlabel('seconds', fontsize=12)
# ax.set_ylabel('PDF', fontsize=12)
# ax.legend()
# plt.show()
# # add interval as new feature
# # check the uniqueness of device and ip
user_per_ip = train[['ip_address', 'user_id']].groupby(['ip_address']).count()
user_per_device = train[['device_id', 'user_id']].groupby(['device_id']).count()
# per_ip_plt = user_per_ip.groupby(['user_id']).size()
# per_device_plt = user_per_ip.groupby(['user_id']).size()
# plt.plot(per_ip_plt.index[1:], per_ip_plt.values[1:])
# plt.plot(per_device_plt.index[1:], per_device_plt.values[1:])
# plt.show()
# # add user_per_ip and user_per_device as new features

# X = X_train[['user_id', 'device_id']]
# group_target = X.columns[-1]
# per_num = X.groupby(group_target).count().reset_index()
# device_num = per_num.rename(columns={X.columns[0]: X.columns[-1] + '_num'})
# X = X.merge(device_num, how='left', on=group_target)
# X = X.iloc[:,-1].to_frame(name=X.columns[-1] + '_num')
############################################################
# Logistic Regression Approach (pipeline + GridSearchCV)
############################################################
##############################
# Build Pipeline
##############################
# Pipeline Stacking
pipeline = Pipeline([
    ('features', DFFeatureUnion([
        ('dates', Pipeline([
            ('extract', ColumnExtractor(DATE_FEATS)),
            ('to_date', DateFormatter()),
            ('to_month', DateToMonth()),
            ('diffs', DateDiffer()),
            ('del', DeleteCol(DATE_FEATS))
        ])),
        ('cat_ohe', Pipeline([
            ('extract', ColumnExtractor(CAT_OHE_FEATS)),
            ('dummy', DummyTransformer())
        ])),
        ('cat_lab', Pipeline([
            ('extract', ColumnExtractor(CAT_LAB_FEATS)),
            ('multi_dummy', LabelTransformer())
        ])),
        ('Add_device_num', Pipeline([
            ('extract', ColumnExtractor(['user_id', 'device_id'])),
            ('groupby_count', AddGroupByCount())
        ])),
        ('Add_ip_address_num', Pipeline([
            ('extract', ColumnExtractor(['user_id', 'ip_address'])),
            ('groupby_count', AddGroupByCount())
        ])),
        ('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer()),
            ('log', Log1pTransformer())
        ]))
    ])),
    ('scale', DFStandardScaler()),
    ('pca', PCA()),
    ('LR', LogisticRegression(random_state=5678))
])
# temp = pipeline.fit(X_train, H2OFrame(y_train.to_frame()))
##############################
# Modeling + Tuning
##############################
# regular
check_params= {
    'pca__n_components': [6, 8],
    'LR__C': [20, 50]
}

for cv in [5]:
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv, scoring='roc_auc')
    create_grid.fit(X_train, y_train)
    print("score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test, y_test)))
    print("!!!!!!!! Best-Fit Parameters From Training Data !!!!!!!!!!!!!!")
    print(create_grid.best_params_)
print("out of the loop")
print("grid best params: ", create_grid.best_params_)
##############################
# Evaluation
##############################
final_model = create_grid.best_estimator_
final_model.fit(X_train, H2OFrame(y_train.to_frame()))

# classification evaluation
from sklearn.metrics import roc_auc_score
y_pred = final_model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_pred)
print(auc_test)

# adjust the recall
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
pr_auc = metrics.auc(recall, precision)
plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0, 1])
# plt.show()

# feature importance
feature_importance = abs(final_model.steps[3][1].coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(final_model.named_steps['features'].get_feature_names())[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()
plt.show()
############################################################
# H2O Approach (pipeline + GridSearchCV)
############################################################
##############################
# Build Pipeline for feature engineering
##############################
# Initialize H2O cluster
h2o.init()
h2o.remove_all()
# Pipeline Stacking
pipeline = Pipeline([
    ('features', DFFeatureUnion([
        ('dates', Pipeline([
            ('extract', ColumnExtractor(DATE_FEATS)),
            ('to_date', DateFormatter()),
            ('to_month', DateToMonth()),
            ('diffs', DateDiffer()),
            ('del', DeleteCol(DATE_FEATS))
        ])),
        ('cat_ohe', Pipeline([
            ('extract', ColumnExtractor(CAT_OHE_FEATS)),
            ('dummy', DummyTransformer())
        ])),
        ('cat_lab', Pipeline([
            ('extract', ColumnExtractor(CAT_LAB_FEATS)),
            ('multi_dummy', LabelTransformer())
        ])),
        ('Add_device_num', Pipeline([
            ('extract', ColumnExtractor(['user_id', 'device_id'])),
            ('groupby_count', AddGroupByCount())
        ])),
        ('Add_ip_address_num', Pipeline([
            ('extract', ColumnExtractor(['user_id', 'ip_address'])),
            ('groupby_count', AddGroupByCount())
        ])),
        ('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer()),
            ('log', Log1pTransformer())
        ]))
    ]))
])
##############################
# Modeling + Tuning
##############################
from h2o.cross_validation import H2OKFold
dataset = pd.concat([X_train, y_train], axis=1)
cv = H2OKFold(dataset, n_folds=5, seed=42)
# H2O approach
("H2OCreator", H2OFrameCreator()),
# ('standardize', H2OScaler()),
# ('pca', H2OPCA()),
('rf', H2ORandomForestEstimator(ntrees=20))

# something new to try
# from scipy.stats import randint
# params = {
#           # "standardize__center":    [True, False],
#           # "standardize__scale":     [True, False],
#           "pca__k":  2,
#               # randint(2, X_train[1:].shape[1]),
#           "rf__ntrees": 20
# # randint(50,60),
#           # "rf__max_depth":          randint(4,8),
#           # "rf__min_rows":           randint(5,10),
#           }
# from h2o.cross_validation import H2OKFold
# from sklearn.model_selection import GridSearchCV
# from h2o.model.regression import h2o_r2_score
# from sklearn.metrics.scorer import make_scorer
# custom_cv = H2OKFold(X_train, n_folds=5, seed=42)
# random_search = GridSearchCV(pipeline, params, cv=custom_cv, scoring='roc_auc', n_jobs=1)
# random_search.fit(X_train, y_train)

# H2O approach
# from h2o.grid.grid_search import H2OGridSearch
# params= {
#     'standardize__center': [True, False],
#     'standardize__scale': [True, False],
#     'pca__K': [2, 3],
#     'rf__ntrees': [10, 20]
# }
# criteria = {"strategy": "RandomDiscrete",
#             "stopping_rounds": 10,
#             "stopping_tolerance": 0.00001,
#             "stopping_metric": "misclassification"}
# grid_search = H2OGridSearch(pipeline,
#                             hyper_params = params,
#                             search_criteria = criteria)






# Shutdown h2o instance
h2o.cluster().shutdown()