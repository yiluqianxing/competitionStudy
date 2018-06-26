import pandas as pd
import numpy as np
#是否白条，分期   加个字段
def findname1(x):
    if x.find(u"分期") !=-1:
        return 1
    if x.find(u"白条") !=-1:
        return 1
    else:
        return 0
def mygroupCount(group11):
    group = group11.ix[group11.sts_order_number == 1]

    global cc, num, bili
    cc += 1
    tmp = np.round(cc / num, 1)
    if tmp == bili:
        bili += 0.1
        print("已完成:", tmp * 100, "%")

    if group.shape[0] == 0:
        #         print("全是退款取消的信息")
        group = group11.ix[group11.sts_order_number == 3]
    count = len(group.no_order_md5.unique())  # 将一个订单的内容合并起来

    # # 统计这个人是否抢中什么，看待运气的心里
    # orderInfo_luck = 0
    # tmp = group.ix[group.sts_order == "未抢中"]
    # if tmp.shape[0] > 0:
    #     orderInfo_luck = tmp.shape[0] / count
    #
    # # 有几个收货人
    # orderInfo_name_rec_md5_num = len(group.name_rec_md5.unique())
    #
    # # 统计是否经常在 上班时间购物
    # orderInfo_work_time_more_cate = (group.orderInfo_work_time.value_counts()
    #     .sort_values(ascending=False).index[0])
    #
    # orderInfo_work_time_more_proportion = (group.orderInfo_work_time.value_counts()
    #                                        .sort_values(ascending=False).iloc[0] / count)
    #
    # # 统计在每年的 多少周购物比较多 orderInfo_week
    # orderInfo_week_more_cate = (group.orderInfo_week.value_counts()
    #     .sort_values(ascending=False).index[0])
    #
    # tmp = group.orderInfo_week.value_counts().sort_values(ascending=False).iloc[0]
    # orderInfo_week_more_proportion = (tmp / count)
    # # 这里时间必须 用.iloc[0]  也是神奇，以后还是都带上这个，免得找不到错误在哪里
    #
    # # 哪一年,月，日，周几购物多
    # orderInfo_year_more_cate = (group.orderInfo_year.value_counts()
    #     .sort_values(ascending=False).index[0])
    # orderInfo_year_more_proportion = (group.orderInfo_year.value_counts()
    #                                   .sort_values(ascending=False).iloc[0] / count)
    #
    # orderInfo_month_more_cate = (group.orderInfo_month.value_counts()
    #     .sort_values(ascending=False).index[0])
    # orderInfo_month_more_proportion = (group.orderInfo_month.value_counts()
    #                                    .sort_values(ascending=False).iloc[0] / count)
    #
    # orderInfo_day_more_cate = (group.orderInfo_day.value_counts()
    #     .sort_values(ascending=False).index[0])
    # orderInfo_day_more_proportion = (group.orderInfo_day.value_counts()
    #                                  .sort_values(ascending=False).iloc[0] / count)
    #
    # orderInfo_isoweekday_more_cate = (group.orderInfo_isoweekday.value_counts()
    #     .sort_values(ascending=False).index[0])
    # orderInfo_isoweekday_more_proportion = (group.orderInfo_isoweekday.value_counts()
    #                                         .sort_values(ascending=False).iloc[0] / count)
    #
    # orderInfo_month_stage_more_cate = (group.orderInfo_month_stage.value_counts()
    #     .sort_values(ascending=False).index[0])
    # orderInfo_month_stage_more_proportion = (group.orderInfo_month_stage.value_counts()
    #                                          .sort_values(ascending=False).iloc[0] / count)
    #
    # # 有多少种支付方式
    # orderInfo_type_pay_cate_num = len(group.type_pay.unique())
    # # 常见支付方式及其比率
    # orderInfo_type_pay_more_cate = (group.type_pay.value_counts()
    #     .sort_values(ascending=False).index[0])
    # orderInfo_type_pay_more_proportion = (group.type_pay.value_counts()
    #                                       .sort_values(ascending=False).iloc[0] / count)
    #
    # # 使用京豆京劵等支付的比例（说明比较活跃）  type_pay_special   1
    # tmp = group.ix[group.type_pay_special == 1]
    # orderInfo_type_pay_special_proportion = tmp.shape[0] / count
    #
    # # 使用白条、分期的比例是多少
    # tmp = group.ix[group.whitePayment == 1]
    # orderInfo_whitePayment_proportion = tmp.shape[0] / count

    # 有多少种订单状态
    orderInfo_sts_order_cate_num = len(group.sts_order.unique())
    # 常见订单状态及其比率
    orderInfo_sts_order_more_cate = (group.sts_order.value_counts()
        .sort_values(ascending=False).index[0])
    orderInfo_sts_order_more_proportion = (group.sts_order.value_counts()
                                           .sort_values(ascending=False)[0] / count)

    # 有多少手机号
    orderInfo_phone_cate_num = len(group.phone.unique())

    # 花费充值单次最大金额，平均每次充值多少,充话费的比例
    tmp = group.ix[group.sts_order == "充值成功"]
    orderInfo_phone_money_max = 0  # 有的人没有充值花费
    orderInfo_phone_money_ave = 0
    if tmp.shape[0] > 0:
        orderInfo_phone_money_max = np.max(tmp.amt_order)
        orderInfo_phone_money_ave = np.mean(tmp.amt_order)
    orderInfo_phone_money_proportion = tmp.shape[0] / count

    # 平均多久购物一次
    tmpA = np.max(group.orderInfo_timestamp)
    tmpB = np.min(group.orderInfo_timestamp)
    orderInfo_buy_time_interval = (tmpA - tmpB) / count

    # 平均每次购物多少钱，购物金额最大的一笔
    orderInfo_buy_money_max = np.max(group.amt_order)
    orderInfo_buy_money_ave = (group.amt_order.sum()) / count
    #最大缺失值，平均缺失值个数
    orderInfo_lossNUm_max = np.max(group.lossNUm)
    orderInfo_lossNUm_ave = (group.lossNUm.sum()) / count

    indexs = {
        "orderInfo_work_time_more_cate": orderInfo_work_time_more_cate,
        "orderInfo_week_more_cate": orderInfo_week_more_cate,
        "orderInfo_year_more_cate": orderInfo_year_more_cate,
        "orderInfo_month_more_cate": orderInfo_month_more_cate,
        "orderInfo_day_more_cate": orderInfo_day_more_cate,
        "orderInfo_isoweekday_more_cate": orderInfo_isoweekday_more_cate,
        "orderInfo_month_stage_more_cate": orderInfo_month_stage_more_cate,
        "orderInfo_type_pay_more_cate": orderInfo_type_pay_more_cate,
        "orderInfo_sts_order_more_cate": orderInfo_sts_order_more_cate,

        "orderInfo_lossNUm_max":orderInfo_lossNUm_max,
        "orderInfo_lossNUm_ave":orderInfo_lossNUm_ave,
        "orderInfo_work_time_more_proportion": orderInfo_work_time_more_proportion,
        "orderInfo_luck": orderInfo_luck,
        "orderInfo_name_rec_md5_num": orderInfo_name_rec_md5_num,
        "orderInfo_week_more_proportion": orderInfo_week_more_proportion,
        "orderInfo_year_more_proportion": orderInfo_year_more_proportion,
        "orderInfo_month_more_proportion": orderInfo_month_more_proportion,
        "orderInfo_day_more_proportion": orderInfo_day_more_proportion,
        "orderInfo_isoweekday_more_proportion": orderInfo_isoweekday_more_proportion,
        "orderInfo_month_stage_more_proportion": orderInfo_month_stage_more_proportion,
        "orderInfo_type_pay_cate_num": orderInfo_type_pay_cate_num,
        "orderInfo_type_pay_more_proportion": orderInfo_type_pay_more_proportion,
        "orderInfo_type_pay_special_proportion": orderInfo_type_pay_special_proportion,
        "orderInfo_whitePayment_proportion": orderInfo_whitePayment_proportion,
        "orderInfo_sts_order_cate_num": orderInfo_sts_order_cate_num,
        "orderInfo_sts_order_more_proportion": orderInfo_sts_order_more_proportion,
        "orderInfo_phone_cate_num": orderInfo_phone_cate_num,
        "orderInfo_phone_money_max": orderInfo_phone_money_max,
        "orderInfo_phone_money_ave": orderInfo_phone_money_ave,
        "orderInfo_phone_money_proportion": orderInfo_phone_money_proportion,
        "orderInfo_buy_time_interval": orderInfo_buy_time_interval,
        "orderInfo_buy_money_max": orderInfo_buy_money_max,
        "orderInfo_buy_money_ave": orderInfo_buy_money_ave
    }

    return pd.Series(data=[indexs[c] for c in indexs], index=[c for c in indexs])



if __name__ == '__main__':
    fulldata = pd.read_csv("../input/fulldata_order_undeal111.csv", low_memory=False)
    # # print(len(fulldata.id.unique()))
    # del fulldata["unit_price"]
    # del fulldata["product_id_md5"]
    # fulldata["whitePayment"] = fulldata.type_pay.apply(lambda x: findname1(x))
    # # print(fulldata.isnull().sum())
    # # print(fulldata.sts_order_number.value_counts())
    # cc = 0
    # num = 167960
    # bili = 0.1
    # tmp122 = fulldata.groupby(by=['id']).apply(mygroupCount)
    # # tmp122 = fulldata.iloc[0:300,:].groupby(by=['id']).apply(mygroupCount)
    # tmp122.insert(0, "id", tmp122.index)
    # tmp122.to_csv("../dealinput/orderCount.csv", index=False)
    # print(tmp122.columns)
    #解决内存过低的问题 https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options
'''
sift 做差分，有库，注意缺失值，乘法还是log，一定要注意 特征工程的细节，相同特征思路，不同的预测分数
每个用户只用完成的数据，以订单为单位，做差分，统计每个订单钱的标准差等
最大值，最小值，最大值与最小值的标准差等无脑堆特征（对于数据量比较大的分类比较好
对于数据量比较小的可能分数会降低）
。 主要是 特征工程视频中的特征。
用户在每年的活跃度，出现次数。
https://blog.csdn.net/u011094454/article/details/78572417

https://zhuanlan.zhihu.com/p/26645088
'''

