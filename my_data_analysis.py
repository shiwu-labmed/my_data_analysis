#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' 辅助快速数据分析的函数 '

__author__ = 'ShiLiu'

#%% 导入依赖包
import pysnooper
import sys
from turtle import position
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit #分层抽样
from IPython.display import display
import os
from pathlib import *
from tqdm import tqdm
import time


#%% 机器学习函数和类
# @pysnooper.snoop('./xgboost optuna调参.log')
class train_test_stratified_split(object):

    def __init__(self, testsize=0.2, ramdom_state=None) -> None:
        self.ss=StratifiedShuffleSplit(
            n_splits=1, test_size=testsize,
            train_size=(1-testsize), random_state=ramdom_state)

    def split(self, x, y):
        # ss.split返回的是生成器，因此必须for in
        x = x.reset_index(drop=True) if type(x)==pd.DataFrame else x
        y = y.reset_index(drop=True) if type(y)==pd.Series else y
        for train_index, test_index in self.ss.split(x, y):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return X_train,X_test,y_train,y_test

ttss = train_test_stratified_split
ttss0_2 = train_test_stratified_split(testsize=0.2, ramdom_state=15)
ttss0_25 = train_test_stratified_split(testsize=0.25, ramdom_state=15)

class MLfeature(object):
    def __init__(self, feature_tab_path) -> None:
        self.__feature_tab_path = feature_tab_path
        self.feature_tab = read_csv8(self.__feature_tab_path)
        self.__feature_tab_copy = self.feature_tab.copy()
        self.__feature_colname = self.feature_tab.columns[0]

    def reload_feature_path(self):
        self.feature_tab = read_csv8(self.__feature_tab_path)
        self.__feature_tab_copy = self.feature_tab.copy()

    def subsets_name(self):
        return self.feature_tab.columns.tolist()

    def open_file(self):
        os.startfile(self.__feature_tab_path)
    
    def __get_subsetname(self, str_containing_subsetname):
        return list(filter(
            lambda x : x in str_containing_subsetname, 
            self.feature_tab.columns.tolist()))\
            [0]

    def __drop_subset_na(self, str_containing_subsetname):
        subsetname = self.__get_subsetname(str_containing_subsetname)
        self.__feature_tab_copy = self.__feature_tab_copy\
                    .dropna(subset=[subsetname])

    def get_subset(self, subsetname):
        self.__drop_subset_na(subsetname)
        subset_list = self.__feature_tab_copy\
            .iloc[:,0]\
            .tolist()
        self.__feature_tab_copy = self.feature_tab.copy()
        return subset_list
        
    def query(self, query_format):
        self.__drop_subset_na(query_format)
        self.__feature_tab_copy = self.__feature_tab_copy.query(query_format)
        return self

    def query_and_get(self, query_format):
        self.query(query_format)
        subsetname = self.__get_subsetname(query_format)
        return self.get_subset(subsetname)

    def add_feature_subset(self, subsetname, subsetlist, importance_list):
        if subsetname in self.subsets_name():
            raise ValueError(
                f' the "{subsetname}" subset already exist, '\
                'please use the "update_feature_subset()" method instead')
        self.feature_tab = pd.merge(
            left=self.feature_tab,
            right=pd.DataFrame({
                self.__feature_colname: subsetlist, 
                subsetname: importance_list
                }),
            on= self.__feature_colname,
            how='left')
        save_csv8(self.feature_tab, self.__feature_tab_path)
    
    def update_feature_subset(self, subsetname, subsetlist, importance_list):
        if subsetname not in self.subsets_name():
            raise ValueError(
                f' the "{subsetname}" subset does not exist, '\
                'please use the "add_feature_subset()" method instead')
        self.feature_tab = self.feature_tab.drop(columns=subsetname)
        self.add_feature_subset(subsetname, subsetlist, importance_list)
os.path.splitext

# bloodcul_MLfeature = MLfeature(
#     'D:\\作业文件\\研究生\\研究生课题\\机器学习血流感染\\基本数据\\'\
#         '变量表原始数据\\血培养机器学习_特征vs重要性.csv')   

#%% 算法函数
def timediff_dedup(df, time_col, timegap=pd.Timedelta('24h')):
    i=0
    # 如果只有一条记录，则i+1超出索引
    dedup_has_been_finished = i+1 >= len(df) 
    while not dedup_has_been_finished:
        in_timediff_bool = True
        while in_timediff_bool and not dedup_has_been_finished:
            timediff_between_i_iplus1 = \
                df.iloc[i+1][time_col] - df.iloc[i][time_col]
            in_timediff_bool = timediff_between_i_iplus1 < timegap
            if in_timediff_bool:
                df = df.drop(df.index[[i+1]])
                # i+1行被删除，如果被删除的是最后一行，则i+1超出了新表索引
                dedup_has_been_finished = i+1 >= len(df)
        i += 1
        #这里必须重新判断是否i+1超出了索引，进行了i发生了变化（i=i+1）
        dedup_has_been_finished = i+1 >= len(df) 
    df['去重后样本数'] = len(df)
    return df.reset_index(drop=True) #原本直接return df 会卡住很长一段时间，但这次没卡很久
    # return df.iloc[0] #甚至这个代码还要跑更久

def format_age(age_str):
    year_match = re.search(r'(\d+)岁', age_str)
    month_match = re.search(r'(\d+)月', age_str)
    day_match = re.search(r'(\d+)天', age_str)

    year = float(year_match.group(1)) if year_match else '0'
    month = float(month_match.group(1) if month_match else '0')
    day = float(day_match.group(1) if day_match else '0')

    return (month*30+day)/365 if year=='0' else year

#%% 全局工具

class maplist(object):

    def __init__(self, iter) -> None:
        self.list = list(iter)

    def __str__(self) -> str:
        return f'maplist object, list:{self.list}'

    __repr__ = __str__

    def map(self, func):
        return maplist(map(func, self.list))

    def filter(self, func):
        return maplist(filter(func, self.list))

class timer_class(object):
    def __init__(self) -> None:
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self, timer_note):
        self.end_time = time.time()
        self.time_consumed = self.end_time - self.start_time
        print(f'{timer_note} 耗时：{self.time_consumed:.30f}')
    
    def stop_and_start(self, timer_note):
        self.stop(timer_note)
        self.start()
    
    stopart = stop_and_start

timer = timer_class()

#%% 测试代码
if __name__ == '__main__':
    html2csv(r'D:\作业文件\研究生\【1】文章\hiv总\实验室检查原始数据\17-22实验室检查')
    html2csv(r'D:\作业文件\研究生\【1】文章\hiv总\实验室检查原始数据\测试')
