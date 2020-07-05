# coding:utf-8
import pandas as pd
import numpy as np
from multiprocessing import cpu_count,Pool

# 计算电脑的cpu有多少核，有几核，就开几个进程
cores = cpu_count()
def parallelize(func,df):
    '''
    对pd.DataFrame格式的数据按核数进行切分，再开多进程处理
    :param func: 处理数据的函数
    :param df: pd.DataFrame格式的数据
    :return:
    '''
    data_split = np.array_split(df,cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(func,data_split))
    pool.close()
    pool.join()
    return data