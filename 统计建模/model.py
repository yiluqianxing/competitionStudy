#coding:utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import numpy as np
import pandas as pd
from scipy.stats import mode
import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from matplotlib import  pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


def tianchong(i,label,data):
    rr = RandomForestRegressor(oob_score=True,random_state=3)
    mm = data.iloc[:,0:i]
#     print(mm.head())
    mm_notnull = mm.ix[mm[label].notnull()]
    mm_isnull = mm.ix[mm[label].isnull()]
    rr.fit(mm_notnull.iloc[:,0:-1],mm_notnull.iloc[:,-1])
    data[label].ix[data[label].isnull()] = rr.predict(mm_isnull.iloc[:,0:-1])
    return data
if __name__ == '__main__':
    fulldata = pd.read_csv("./basicOk.csv")

    #其他表
    cd_ln = pd.read_csv("./content_ext_crd_cd_lnok.csv")
    fulldata = pd.read_csv("./basicOk.csv")
    fulldata = fulldata.merge(cd_ln, how='left', left_on='REPORT_ID', right_on='REPORT_ID')
    label = fulldata.columns
    #是否用随机森林填充                                                          的点点滴滴的点点滴滴端到端
    for i in range(1, len(label)):
        if fulldata[label[i]].isnull().any():
            print("填充缺失值")
            fulldata = tianchong(i + 1, label[i], fulldata)
    #是否用0填充
    fulldata.fillda(0,inplace = True)
    print(fulldata.isnull().sum().sort_values(ascending=False))

    # 获取训练集
    basic = pd.read_csv("../input/contest_basic_train.tsv", sep='\t')
    mm = pd.concat([basic['REPORT_ID'], basic['Y']], axis=1)
    train = mm.merge(fulldata, how='left', left_on='REPORT_ID', right_on='REPORT_ID')
    print(train.isnull().sum().sort_values(ascending=False))
    print(train.head())
    #获取训练集
    trainx = train.iloc[:, 2:]
    trainy = train.iloc[:, 1]
    print(trainx.shape, trainy.shape)
    print("trainx.shape", trainx.shape)
    print("trainy.shape", trainy.shape)
    from sklearn.cross_validation import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(trainx, trainy, test_size=0.3, random_state=0)
    print("x_train.shape", x_train.shape)
    print("x_valid.shape", x_valid.shape)
    print("y_train.shape", y_train.shape)
    print("y_valid.shape", y_valid.shape)
    #获取AUC
    skf = StratifiedKFold(n_splits=5)
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        StratifiedKFold=True,
        seed=27)
    xgb_param = xgb1.get_xgb_params()
    # 构建稀疏矩阵，运行更快
    xgtrain = xgb.DMatrix(x_train, label=y_train)
    # xgtest =xgb.DMatrix(dtest[predictors].values)
    mm = xgb.cv(xgb_param, xgtrain, num_boost_round=1000, nfold=5,
                metrics='auc', early_stopping_rounds=50)
    print(mm[-3:])