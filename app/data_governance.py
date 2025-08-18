#%%
if __name__=='__main__':   
    __package__ = 'my_data_analysis.app'

import pandas as pd 
import numpy as np
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz

from ..base_l2 import Simplified_pandas  as spd 
from ..base_l1 import datafile_operate  as dfo 

#%% 文本相似性去重
def text_similarity_deduplicate(
        arr:np.array,
        threshold:float=80,
        keep:float=2,
        similarity_func=lambda t1,t2: fuzz.ratio(t1,t2), 
        ):
    fpf_similarity_func = np.frompyfunc(similarity_func, 2, 1)
    keeped_x_ind = list(range(keep))
    for x_ind in tqdm(range(keep,len(arr))):
        keeped_x = arr[keeped_x_ind]
        x2check = arr[x_ind]
        similarities = fpf_similarity_func(x2check, keeped_x)
        max_similarities = np.partition(similarities,keep)[-keep:] \
            if len(similarities)>keep else similarities
        if np.min(max_similarities) <= threshold:
            keeped_x_ind.append(x_ind)
    # with tqdm(total=len(arr)) as pbar:
        # i=1
        # while i<len(se):
        #     similarity_se = se[0:i].map(lambda x: similarity_func(x,se[i]))
        #     if max(similarity_se) > threshold:
        #         se = se.drop(i).reset_index(drop=True)
        #     else:
        #         i = i+1
        #     pbar.update()
    return keeped_x_ind

if __name__=='__main__':
    # SequenceMatcher(None, '今天中午吃什么','今天中午什么都不想吃').quick_ratio()
    # SequenceMatcher(None, '今天中午吃什么','午今么天吃想什都不中').quick_ratio()
    # fuzz.ratio('今天中午吃什么','今天中午什么都不想吃')
    # fuzz.ratio('今天中午吃什么','午今么天吃想什都不中')
    # fuzz.ratio('今天','今')
    testpath = Path(r'E:/工作相关/项目·科研/20230607硕士毕业论文/基本数据/血培养原始数据21到23')
    uncategorize_note_data = dfo.csv.read8(testpath/'21到23年血培养_去掉_非血培养_标本_初筛')\
        .query('~标本备注.isna()')\
        .loc[:,['标本备注','血备注校正']]\
        .reset_index(drop=True)
    uncategorize_note_data\
        .iloc[text_similarity_deduplicate(uncategorize_note_data['标本备注'], 75)]\
        .to_csv(testpath/'不相似待分类血培养备注.csv',index=False)
    
#%% knn模型分类文本
def knn_categorize_text(
        text:str, 
        trainX:'np.array|pd.Series', trainy:'np.array|pd.Series',
        k=3,
        similarity_func=lambda t1,t2: fuzz.ratio(t1,t2), 
        ):
    trainX,trainy = (
        se.values for se in (trainX,trainy) if isinstance(se, pd.Series))
    fpf_similarity_func = np.frompyfunc(similarity_func, 2, 1)
    similarities = fpf_similarity_func(text, trainX)
    max_simi_loc = np.argmax(similarities)
    ymax,xmax = trainy[max_simi_loc],trainX[max_simi_loc]
    ys = trainy[np.argpartition(similarities,k)[-k:]]
    ys_unique,ys_counts = np.unique(ys, return_counts=True)
    ymost = ys_unique[np.argmax(ys_counts)]
    return ymax if text==xmax else ymost

if __name__=='__main__':
    # knn_categorize_text('1', np.array(list('123124')), np.array(list('123124')))
    testpath = Path(r'E:/工作相关/项目·科研/20230607硕士毕业论文/基本数据/血培养原始数据21到23')
    dissimilar_notes = dfo.csv.read8(testpath/'不相似待分类血培养备注.csv')
    costum_note_cate = dfo.csv.read8(testpath/'特殊备注自定义处理.csv')
    dissimilar_notes['血备注校正'] = dissimilar_notes['标本备注']\
        .map(lambda x: 
             knn_categorize_text(
                 x, costum_note_cate['备注'], costum_note_cate['标本类型大类']))
    dissimilar_notes.columns = ['key', 'value']
    pd.concat([costum_note_cate, dissimilar_notes])\
        .to_csv(testpath/'特殊备注自定义处理1.csv',index=False)

#%% 将数字列增加英文单引号（防止excel自动转数字）
def add1quot4digit(x)->str:
    return f"'{x:.0f}" if isinstance(x,(int,float)) else f"'{x}"

#%% 新建年季月列
def format_year(timestr:str):
    time = pd.to_datetime(
        timestr if isinstance(timestr, str) else str(timestr)
        ,errors='coerce')
    return f'{time.year}年'

def format_quarter(timestr:str):
    time = pd.to_datetime(
        timestr if isinstance(timestr, str) else str(timestr)
        ,errors='coerce')
    return f'{time.year}年 {time.quarter:02d}季度'

def format_month(timestr:str):
    time = pd.to_datetime(
        timestr if isinstance(timestr, str) else str(timestr)
        ,errors='coerce')
    return f'{time.year}年 {time.month:02d}月'

def format_month_scale(timestr:str, m_start:int, m_end:int):
    time = pd.to_datetime(
        timestr if isinstance(timestr, str) else str(timestr)
        ,errors='coerce')
    if m_start == m_end:
        raise ValueError("m_start cannot be equal to m_end")
    
    if m_start<m_end:
        result = f'{time.year}年{m_start}月到{m_end}月' \
            if m_start<=time.month<=m_end \
            else f'{time.year}年其他时间'
    elif m_start>m_end:
        if time.month>m_start:
            result = f'{time.year}年{m_start}月到{time.year+1}年{m_end}月'
        elif time.month<m_end:
            result = f'{time.year-1}年{m_start}月到{time.year}年{m_end}月'
        else:
            result = f'{time.year}年{m_end+1}月到{m_start-1}月'
    return result

def add_year_quarter_month_col(
        df:pd.DataFrame, timecol:str, 
        col_prefix:str=None, fillnaval:str='2200-01-01') -> pd.DataFrame:
    if df[timecol].dtype != "datetime64[ns]":
        df[timecol] = pd.to_datetime(df[timecol].map(str), errors='coerce')
    df[timecol] = df[timecol].fillna(pd.to_datetime(fillnaval))
    col_prefix = timecol if col_prefix is None else col_prefix
    df[f'{col_prefix}年'] = df[timecol].dt.year.map(int).map(str)+'年'
    df[f'{col_prefix}季'] = df[f'{col_prefix}年']+' '+\
        df[timecol].dt.quarter.map(int).map(str).str.zfill(2)+'季'
    df[f'{col_prefix}月'] = df[f'{col_prefix}年']+' '+\
        df[timecol].dt.month.map(int).map(str).str.zfill(2)+'月'
    return df

def add_year_quarter_month_cols(
        df:pd.DataFrame, timecols:list, 
        col_prefixs:list=[], fillnaval:str='2200-01-01') -> pd.DataFrame:
    col_prefixs.append([None]*(len(timecols)-len(col_prefixs)))
    for col,prefix in zip(timecols, col_prefixs):
        df = add_year_quarter_month_col(df, col, prefix, fillnaval)
    return df
