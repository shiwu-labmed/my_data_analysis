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

#%%
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
    
#%%
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

 