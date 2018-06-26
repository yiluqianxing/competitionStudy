from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
train =  pd.read_csv("../trainFullNumberNew88.csv",encoding = "utf-8",header = None)
validata =  pd.read_csv("../validataFullNumberNew88.csv",encoding = "utf-8",header = None)
print(train.head())
print(".............")
l = 56  #55 维
trainx = train.iloc[:,1:l]
trainy = train.iloc[:,0]

validatax = validata.iloc[:,1:l]
validatay = validata.iloc[:,0]

trainx.fillna(-888,inplace = True)
validatax.fillna(-888,inplace = True)
trainx.isnull().sum(),validatax.isnull().sum()

trainx[trainx == -888] = 3
trainx[trainx == -666] = 4
trainx[trainx == 8] = 5

validatax[validatax == -888] = 3
validatax[validatax == -666] = 4
validatax[validatax == 8] = 5

xgb1 = XGBClassifier(
    #         booster = 'dart',
            learning_rate =0.1,
            n_estimators=100,
            max_depth=5,
            min_child_weight=5,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            nthread=3,
            scale_pos_weight=1,
            StratifiedKFold = True,          #0.866124	0.008005	0.965337	0.002286
            seed=1)
#     xgb1.fit(trainx,trainy)
#     prexgb = xgb1.predict(validatax)
xgb1.fit(trainx,trainy)
print(trainx.shape,trainy.shape,validatax.shape,validatay.shape)
prexgb = xgb1.predict(validatax)
aa = accuracy_score(prexgb,validatay)
print("acc",aa)
cc = f1_score (prexgb,validatay, average='macro')
print("f1",cc)
print(f1_score (prexgb,validatay, average=None))

xgb_params = xgb1.get_xgb_params()
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(trainx, trainy)
dtrain_split = xgb.DMatrix(x_train_split, label=y_train_split)
dtest_split = xgb.DMatrix(x_test_split)

res = xgb.cv(xgb_params, dtrain_split, num_boost_round=1000, nfold=3, seed=1, stratified=True,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)  #metrics='macro',

best_nrounds = res.shape[0] - 1
print(np.shape(x_train_split), np.shape(x_test_split), np.shape(y_train_split), np.shape(y_test_split))
gbdt = xgb.train(xgb_params, dtrain_split, best_nrounds)
y_predicted = gbdt.predict(dtest_split)