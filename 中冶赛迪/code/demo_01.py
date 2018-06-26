import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score 
train =  pd.read_csv("../trainFullNumberNew8.csv",encoding = "utf-8",header = None)
validata =  pd.read_csv("../validataFullNumberNew8.csv",encoding = "utf-8",header = None)
# print(train.head())
print(".............")
aa = []
f1_mean = []
f1_eve = []
l = 100
for i in range(30,l):
    trainx = train.iloc[:,1:i]
    trainy = train.iloc[:,0]

    validatax = validata.iloc[:,1:i]
    validatay = validata.iloc[:,0]
    trainx.fillna(-888, inplace=True)
    validatax.fillna(-888, inplace=True)
    trainx[trainx == -888] = 3
    trainx[trainx == -666] = 4
    trainx[trainx == 8] = 5
    validatax[validatax == -888] = 3
    validatax[validatax == -666] = 4
    validatax[validatax == 8] = 5
    xgb1 = XGBClassifier(
        #         booster = 'dart',
        learning_rate=0.1,
        #         n_estimators=mm.shpa[0],
        max_depth=5,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        #         objective= 'binary:logistic',
        nthread=3,
        scale_pos_weight=1,
        StratifiedKFold=True,  # 0.866124	0.008005	0.965337	0.002286
        seed=1)
    #     xgb1.fit(trainx,trainy)
    #     prexgb = xgb1.predict(validatax)

    xgb1.fit(trainx, trainy)
    print(trainx.shape, trainy.shape, validatax.shape, validatay.shape)
    print(".................")
    print("i:",i)
    prexgb = xgb1.predict(validatax)
    aa.append(accuracy_score(prexgb, validatay))
    print("acc", aa)
    f1_mean.append(f1_score(prexgb, validatay, average='micro'))
    print("f1_mean", f1_mean)
    f1_eve.append(f1_score(prexgb, validatay, average=None))
x = np.arange(10,l)
plt.figure()
plt.subplot(2,3,1)
plt.plot(x,aa)
plt.ylabel("accuracy")

plt.subplot(2,3,2)
plt.plot(x,f1_mean)
plt.ylabel("f1_micro")

f1_eve = pd.DataFrame(f1_eve)
plt.subplot(2,3,3)
plt.plot(x,f1_eve[0])
plt.ylabel("f1_label 1")

plt.subplot(2,3,4)
plt.plot(x,f1_eve[1])
plt.ylabel("f1_label 2")

plt.subplot(2,3,5)
plt.plot(x,f1_eve[2])
plt.ylabel("f1_label 3")

plt.subplot(2,3,6)
plt.plot(x,f1_eve[3])
plt.ylabel("f1_label 4")
plt.tight_layout()
plt.show()


