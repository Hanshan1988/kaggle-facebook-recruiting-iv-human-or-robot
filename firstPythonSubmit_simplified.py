# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:39:15 2015

@author: Hans
"""

# http://stackoverflow.com/questions/18885175/read-a-zipped-file-as-a-pandas-dataframe
# http://stackoverflow.com/questions/19371860/python-open-file-from-zip-without-temporary-extracting-it

from __future__ import division
import pandas as pd
import zipfile as zf
import numpy as np
#from ggplot import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')

z = zf.ZipFile('../input/bids.csv.zip', 'r')
df = pd.read_csv(z.open('bids.csv'))

z= zf.ZipFile('../input/train.csv.zip', 'r')
df_train = pd.read_csv(z.open('train.csv'))

z= zf.ZipFile('../input/test.csv.zip', 'r')
df_test = pd.read_csv(z.open('test.csv'))

# The bids table has much more unqiue bidder_id than the bidders table
print len(df["bidder_id"].unique())
# 6614
print len(df_train["bidder_id"].unique())
# 2013
print len(df_test['bidder_id'].unique())
# 4700
# Some bids refers to bidders in the test set

# Try separating the groups
df['group'] = np.where(df['time'] > 9.74e15, 1, np.where(df['time'] < 9.68e15, 2, 3))

# Left join, because some bids refer to bidders in the test set
df2 = df.merge(df_train[['bidder_id','outcome']], on = 'bidder_id', how = 'left').fillna(-1)
len(df)==len(df2)

df2["outcome"].hist()

df2.columns.tolist()

df2[df2["outcome"] == 1].describe()
len(df2["bidder_id"].unique())
# 6614

print len(df2[df2.group==1].bidder_id.unique()) #2954
print len(df2[df2.group==2].bidder_id.unique()) #3016
print len(df2[df2.group==3].bidder_id.unique()) #3014
# Thus many bidders are across more than one time group

def getUnique(df, cols):
    for col in cols:
        numUniq = len(df[col].unique())
        print('Column ' + col + ':' + str(numUniq))

getUnique(df2, df2.columns.tolist())

df2['device2'] = df2['device'].str[:5]
# Device are all phones
df2[df2.bidder_id =='e90e4701234b13b7a233a86967436806wqqw4']['merchandise'].value_counts()
# Only one bidder has more than one merchandise

# Aggregate on an auction level for inter-response time
df2 = df2.sort_values(['time'])
grouped_auction = df2.groupby('auction', as_index = False)
len(grouped_auction)
# 15051 auctions
# grouped_auction = grouped_auction.sort(['time'])
auction_sizes = grouped_auction['bid_id'].agg({'group_size' : lambda x: x.nunique()})
auction_sizes.sort_values('group_size')
auction_sizes['group_size'].value_counts().head() #1096 auctions with only 1 bid
one_bid_list = auction_sizes.sort('group_size').auction.tolist()[0:1097]
df2[df2.auction.isin(one_bid_list)].head(20)

resp_time_auct = grouped_auction['time'].transform(lambda x: (x - x.shift()).fillna(-10))
df2_resp_time_auct = pd.concat([df2, resp_time_auct], axis = 1)
df2_resp_time_auct['grp_count'] = grouped_auction.cumcount()
colnames = df2_resp_time_auct.columns.tolist()
colnames[-2] = 'resp_time_auct'
df2_resp_time_auct.columns = colnames
# Possibly also add time between bids for the same bidder same auction

# df2_resp_time.sort(['auction', 'time'])[['bidder_id', 'auction', 'time', 'resp_time_auct', 'outcome']]

# Unique bidders
len(df2_resp_time_auct[df2_resp_time_auct['outcome']==0].bidder_id.unique())
# 1881
len(df2_resp_time_auct[df2_resp_time_auct['outcome']==1].bidder_id.unique())
# 103

# How many auctions are initiated by robots vs humans
df2_init = df2_resp_time_auct[df2_resp_time_auct['resp_time']==-10]
df2_init_outcome = df2_init['outcome'].value_counts()
# Total auctions attended by robots vs humans
df2_resp_time_auct['outcome'].value_counts()
# Percentage i
for i in range(3):
    print str(i) + ': ' + str(df2_init_outcome.iloc[i]/df2_resp_time['outcome'].value_counts().iloc[i])

# Aggregate on bidder level, inter-response time on own bids for different auctions
grouped_bidder = df2.groupby('bidder_id', as_index = False)
len(grouped_bidder)
# 6614 bidders
resp_time_bidder = grouped_bidder['time'].transform(lambda x: (x - x.shift()).fillna(-10))
df2_resp_time = pd.concat([df2_resp_time_auct, resp_time_bidder], axis = 1)
colnames = df2_resp_time.columns.tolist()
colnames[-1] = 'resp_time_bidder'
df2_resp_time.columns = colnames

# Aggregate on a bidder level
# df2_resp_time2 = df2_resp_time[df2_resp_time['resp_time']>=0]
df2_resp_time2 = df2_resp_time
grouped_id = df2_resp_time2.groupby(['bidder_id'])
grouped_resp = grouped_id['resp_time_auct'].agg({'mean_resp_a':np.mean, 'min_resp_a':np.min,
                                                 'max_resp_a':np.max, 'median_resp_a':np.median,
                                                 'num_inst_resp_a':lambda x: x.tolist().count(0),
                                                 'num_first_a':lambda x: x.tolist().count(-10)})
grouped_resp2 = grouped_id['resp_time_bidder'].agg({'mean_resp_b':np.mean, 'min_resp_b':np.min,
                                                    'max_resp_b':np.max, 'median_resp_b':np.median,
                                                    'num_inst_resp_b':lambda x: x.tolist().count(0)})
grouped_time = grouped_id['time'].agg({'num_u_times' : lambda x: x.nunique()})                                        
grouped_num_bids = grouped_id['bid_id'].agg({'num_bids':lambda x: x.count()})
grouped_auct = grouped_id['auction'].agg({'num_u_auct' : lambda x: x.nunique()})
grouped_merch = grouped_id['merchandise'].agg({'num_u_merch' : lambda x: x.nunique()})
grouped_device = grouped_id['device'].agg({'num_u_device' : lambda x: x.nunique()})
grouped_country = grouped_id['country'].agg({'num_u_country' : lambda x: x.nunique()})
grouped_ip = grouped_id['ip'].agg({'num_u_ip' : lambda x: x.nunique()})
grouped_url = grouped_id['url'].agg({'num_u_url' : lambda x: x.nunique()})
grouped_outcome = grouped_id['outcome'].agg({'outcome' : np.max})

df2_summary = pd.concat([grouped_resp, grouped_resp2, grouped_time, grouped_num_bids, 
                         grouped_auct, grouped_merch, grouped_device, grouped_country,
                         grouped_ip, grouped_url, grouped_outcome], axis = 1)
df2_summary = df2_summary.reset_index()

df2_summary.isnull().sum(axis=0)
len(df2_summary) # 6614
df2_summary['num_bids'].hist(bins =range(0, 50, 1))
df2_summary['p_inst_resp_a'] = df2_summary['num_inst_resp_a'] / df2_summary['num_bids'].astype(float)
df2_summary['p_inst_resp_b'] = df2_summary['num_inst_resp_b'] / df2_summary['num_bids'].astype(float)
# Write to file
df2_summary.to_csv('df2_summary.csv')

# More EDA
low_bids = df2_summary[df2_summary['num_bids'] ==2]
for i in range(6,14):
    print low_bids.iloc[:,i].unique().tolist()    

#
#Split out merch across the columns
cols_to_retain = ['merch']
enc_df = df2_summary[cols_to_retain]
enc_dict = enc_df.T.to_dict().values()
from sklearn.feature_extraction import DictVectorizer as DV

vectorizer = DV(sparse = False)
vec_enc = vectorizer.fit_transform(enc_dict)

# Mean bids per auction
by_bidder_auct = df2_resp_time2.groupby(['bidder_id', 'auction'])
bids_by_auct = by_bidder_auct['bid_id'].agg(lambda x: x.count()).reset_index()
mean_bids_per_auct = \
    bids_by_auct.groupby('bidder_id')['bid_id'].agg({'mean_bids_per_auct':np.mean})
df2_summary = \
    pd.merge(df2_summary, mean_bids_per_auct.reset_index(), how='left', on='bidder_id')

def aggBidsPerVarPerBidder(agg, var, df):
    # Agg is np.mean, np.median, np.min, np.max
    by_bidder_var = df.groupby(['bidder_id', var])
    bids_by_var =  by_bidder_var['bid_id'].agg(lambda x: x.count()).reset_index()
    agg_bids_per_var = \
        bids_by_var.groupby('bidder_id')['bid_id'].agg({'agg_bids_per_' + var:agg}).reset_index()
    return agg_bids_per_var

# Mean bids per device
mean_bids_per_auct = aggBidsPerVarPerBidder(np.mean, 'auction', df2_resp_time2)
df2_summary = \
    pd.merge(df2_summary, mean_bids_per_auct, how='left', on='bidder_id')

# Max number of bids that share the same left-most-4, 5, 6 digits of time for each bidder
for i in range(4, 7):
    df2_resp_time2['time_left{}'.format(i)] = df2_resp_time2.time.astype(str).str[:i]
max_bids_per_timeleft4 = aggBidsPerVarPerBidder(np.max, 'time_left4', df2_resp_time2)
max_bids_per_timeleft5 = aggBidsPerVarPerBidder(np.max, 'time_left5', df2_resp_time2)
max_bids_per_timeleft6 = aggBidsPerVarPerBidder(np.max, 'time_left6', df2_resp_time2)
max_bids_per_time = pd.concat([max_bids_per_timeleft4, max_bids_per_timeleft5, 
                               max_bids_per_timeleft6], axis=1)
max_bids_per_time = max_bids_per_time.T.drop_duplicates().T
df2_summary = \
    pd.merge(df2_summary, max_bids_per_time, how='left', on='bidder_id')

# Look only at training data
df2_train = df2_summary[df2_summary['outcome'] >= 0]
df2_test = df2_summary[df2_summary['outcome'] < 0]

########################################################################

# Modelling
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

df2_train.columns
features = ['num_inst_resp_a', 'mean_resp_a', 'min_resp_a', 'median_resp_a', 'num_bids', 
            'num_u_auct', 'num_u_device', 'num_u_country', 'num_first_a', 'num_u_times',
            'num_u_ip', 'num_u_url', 'mean_bids_per_auct', 'num_inst_resp_b',
            'mean_resp_b', 'median_resp_b', 'min_resp_b', 'agg_bids_per_time_left4',
            'agg_bids_per_time_left5', 'agg_bids_per_time_left6']
# Looked at feature importance and removed min_resp_b
features = ['num_inst_resp_a', 'mean_resp_a', 'median_resp_a', 'num_bids', 
            'num_u_auct', 'num_u_device', 'num_u_country', 'num_first_a', 'num_u_times',
            'num_u_ip', 'num_u_url', 'mean_bids_per_auct', 'num_inst_resp_b',
            'mean_resp_b', 'median_resp_b', 'agg_bids_per_time_left4',
            'agg_bids_per_time_left5', 'min_resp_a', 'agg_bids_per_time_left6']
X = df2_train[features].as_matrix()
y = df2_train['outcome'].as_matrix()

# For use to compete
X_test = df2_test[features].as_matrix()

##################################################################
# K-fold cross-validation
# Use stratified K-fold for imbalanced class
from sklearn.cross_validation import StratifiedKFold
def run_cv(X, y, predict_proba, clf_class, **kwargs):
    # Construct a kfolds object
    skf = StratifiedKFold(y, n_folds=5, shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        if predict_proba == False:
            y_pred[test_index] = clf.predict(X_test)
        else:
            y_pred[test_index] = clf.predict_proba(X_test)[:, 1].astype(float)
    return y_pred
    
def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC

print "GBM:"
print "%.3f" % accuracy(y, run_cv(X, y, False, GBC, n_estimators=200)) 
gbm_performance = roc_auc_score(y, run_cv(X, y, True, GBC, n_estimators=200))
print 'GBM: Area under the ROC curve = {}'.format(gbm_performance)

print "Random forest:"
print "%.3f" % accuracy(y, run_cv(X, y, False, RF, n_estimators=200))
rf_performance = roc_auc_score(y, run_cv(X, y, True, RF, n_estimators=200))
print 'RF: Area under the ROC curve = {}'.format(rf_performance)

importances = pd.Series(gbm.feature_importances_, index=df2_train[features].columns)
print importances.order(ascending=False)[:20]

###############################################################################
# Implementing model
df2_test.reset_index(inplace=True)

# Try RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, min_samples_split=5)
rf.fit(X, y)
rf_preds = rf.predict_proba(X_test)[:, 1].astype(float)
test = pd.concat((df2_test['bidder_id'],pd.DataFrame(rf_preds)), axis=1)
test = test.rename(columns={'bidder_id': 'bidder_id', 0: 'prediction'})
test2 = pd.merge(df_test, test, how='left').fillna(0)
test2[['bidder_id', 'prediction']].reset_index(drop=True).to_csv('../output/rf_test_file.csv', index=False)
