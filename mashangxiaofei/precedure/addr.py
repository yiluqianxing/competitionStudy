import pandas as pd
import numpy as np


def mygroup(group):
    global n,c,d
    c += 1
    if np.round(c/d,1) == n:
        print("已完成：",c/d)
        n += 0.1
    addr_num = len(group.region.unique())
    group = group.region
    addr_more_cate = group.value_counts().sort_values(ascending=False).index[0]
    addr_more_proportion = group.value_counts().sort_values(ascending=False).iloc[0] / addr_num
    addr_less_cate = group.value_counts().sort_values().index[0]
    addr_less_proportion = group.value_counts().sort_values().iloc[0] / addr_num
    indexs = {
        "addr_more_cate": addr_more_cate,
        "addr_less_cate": addr_less_cate,
        "addr_more_proportion": addr_more_proportion,
        "addr_less_proportion": addr_less_proportion,
        "addr_num": addr_num
    }
    return pd.Series(data=[indexs[c] for c in indexs], index=[c for c in indexs])

if __name__ == '__main__':
    train = pd.read_csv("../input/train_recieve_addr_info.csv")
    test = pd.read_csv("../input/test_recieve_addr_info.csv")
    fulldata = pd.concat([train,test],axis = 0,ignore_index = True)

    mm = fulldata.region.value_counts().sort_values(ascending = False).index[0]
    fulldata.region.fillna(mm,inplace = True)

    n = 0.1
    c = 0
    d = fulldata.shape[0]
    tmp1 = fulldata.groupby(by=['id']).apply(mygroup)
    tmp1.insert(0, "id", tmp1.index)
    print(tmp1.isnull().sum())
    tmp1.to_csv("../dealinput/addr_infoCount.csv",index = False)