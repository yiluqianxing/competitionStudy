import pandas as pd
import numpy as np


def findname(x):
    if x.find(u"混合") != -1:
        return 1
    if x.find(u"京豆") != -1:
        return 1
    if x.find(u"京劵") != -1:
        return 1
    if x.find(u"东券") != -1:
        return 1
    if x.find(u"余额") != -1:
        return 1
    else:
        return 0


def tranf(x):
    x = str(x)
    if x.startswith(('完成', '充值成功', '已完成', '出票成功', '已收货', '已晒单', "过期关闭",
                    '付款成功', '购买成功', '缴费成功', '已确认', '已入住', '过期放弃',
                     '部分充值成功;退款成功', '未入住', '未抢中', '预约完成')):
        x = 1
    elif x.startswith((
            '等待收货', '等待付款', '发货中',
            '预订中', '等待分期付款', '正在充值',
            '正在送货（暂不能上门自提）', '等待付款确认',
            '正在出库', '等待厂商处理', '等待发码', '等待揭晓',
            '请上门自提', '商品出库')):
        x = 2

    else:
        x = 3
    return x


import time


def gettime7(x):  # 修改时间格式
    x = str(x)
    tmp = x.split("-")
    if len(tmp) < 2:
        if len(x) > 10:
            x = x[0:10]
        x = int(x)
        time_local = time.localtime(x)
        return time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    else:
        return x


def gettime8(x):
    t = time.strptime(x, '%Y-%m-%d %H:%M:%S')  # 秒的话加上秒
    t = int(time.mktime(t))
    return t


def gettime9(x):
    x = str(x)
    tmp = x.split("-")[1]
    print(tmp)


import arrow
# arrow  时间函数，可以对时间进行统一编码

def parse_date(date_str, str_format='YYYY-MM-DD HH:mm:ss'):
    d = arrow.get(date_str, str_format)
    # 月初，月中，月末
    month_stage = int((d.day - 1) / 10) + 1
    if month_stage == 4:
        month_stage = 3

    work_time = 0
    if (d.hour < 12 and d.hour > 8) or (d.hour < 18 and d.hour > 13):
        work_time = 1  # 1表示上班时间在购物
    return (d.timestamp, d.year, d.month, d.day, d.week, d.isoweekday(), month_stage, work_time)


def parse_time_order(date):
    d = parse_date(date, 'YYYY-MM-DD HH:mm:ss')
    return pd.Series(d,
                     index=['orderInfo_timestamp', 'orderInfo_year', 'orderInfo_month',
                            'orderInfo_day', 'orderInfo_week', 'orderInfo_isoweekday',
                            'orderInfo_month_stage', 'orderInfo_work_time'],
                     dtype=np.int32)

if __name__ == '__main__':
    train = pd.read_csv("../input/train_order_info.csv")
    test = pd.read_csv("../input/test_order_info.csv")
    fulldata = pd.concat([train, test], axis=0, ignore_index=True)
    # 先填充缺失值
    cate_feature = ['type_pay', 'phone',
                    'no_order_md5', 'name_rec_md5']
    for i in range(len(cate_feature)):
        #     print(fulldata[cate_feature[i]].value_counts().sort_values(ascending = False).index[0])
        fulldata[cate_feature[i]].fillna(fulldata[cate_feature[i]].value_counts().sort_values(ascending=False).index[0],
                                         inplace=True)

    fulldata['amt_order'].fillna(fulldata['amt_order'].median(), inplace=True)
    fulldata['amt_order'].value_counts().sort_values(ascending=False)
    ulimit = np.percentile(fulldata['amt_order'], 99)
    fulldata['amt_order'].loc[fulldata['amt_order'] > ulimit] = ulimit
    fulldata['amt_order'] = fulldata['amt_order'].apply(lambda x: abs(x))
    np.min(fulldata['amt_order']), np.max(fulldata['amt_order'])
    fulldata["lossNUm"] = fulldata.isnull().sum(axis=1)
    # 先把状态填充，找出有多少状态取消，多少状态
    fulldata["sts_order_number"] = 0
    fulldata["type_pay_special"] = 0
    fulldata["sts_order"].fillna(fulldata["sts_order"].value_counts().sort_values(ascending=False).index[0],
                                 inplace=True)
    fulldata["sts_order_number"] = fulldata["sts_order"].apply(lambda x: tranf(x))
    fulldata["type_pay"] = fulldata["type_pay"].astype("str")
    fulldata["type_pay_special"] = fulldata["type_pay"].apply(lambda x: findname(x))
    print(len(fulldata.id.unique()))
    print("缺失值填充完毕，开始进行时间处理")
    fulldata.time_order[fulldata[fulldata.time_order == "0"].index] = "2017-05-31 21:48:13"
    # fulldata.time_order.value_counts().sort_values(ascending = False)
    fulldata["time_order"].fillna(fulldata["time_order"].value_counts().sort_values(ascending=False).index[0],
                                  inplace=True)
    # 拆分时间，月份  1——10，11——20，21——31    时间  上下午工作时间，非工作时间
    fulldata.time_order = fulldata.time_order.apply(lambda x: gettime7(x))

    dateInfo = fulldata.time_order.apply(lambda x: parse_time_order(x))
    fulldata = pd.concat([fulldata, dateInfo], axis=1)
    print("时间处理完成。。。。。")
    print("id.unique()=",len(fulldata.id.unique()))
    fulldata.to_csv("../input/fulldata_order_undeal111.csv", index=False)
