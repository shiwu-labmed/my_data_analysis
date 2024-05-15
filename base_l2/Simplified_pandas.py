
#%% 导入包
import pandas as pd 
import numpy as np 
import tqdm
import os
from IPython.display import display

import win32con
import win32gui
import win32com.client
from pathlib import *
Path.cwd()

from ..base_l1 import datafile_operate as dfo 

shell = win32com.client.Dispatch("WScript.Shell")


#%% region dataframe操作
def col2date(df,lst,errors_='raise',normalize=False):
    for date_col in lst:
        df[date_col] = pd.to_datetime(df[date_col], errors=errors_)
        if normalize:
            df[date_col] = df[date_col].dt.normalize()

def col2num(df,lst,errors_='raise'):
    for num_col in lst:
        df[num_col] = pd.to_numeric(df[num_col], errors=errors_)

class move_col(object):
    def __init__(self, *col_s) -> None:
        self.moved_col = col_s[::-1]

    def indf(self, df):
        self.df = df
        self.columns = df.columns.tolist()
        return self

    def before(self, col):
        position = self.columns.index(col)-1
        return self.to_pos(position)

    def after(self, col):
        position = self.columns.index(col)
        return self.to_pos(position)

    def to_pos(self, position):
        # [self.df.insert(position, c, self.df.pop(c)) for c in self.moved_col]
        for c in self.moved_col:
            self.columns.remove(c)
            self.columns.insert(position, c)
        return self.df.reindex(columns=self.columns)
    
def rand_df():
    return pd.DataFrame(np.random.rand(3,4), columns=list('ABCD'))
#endregion

#region 展示dataframe信息
def print_col(df,num=None):
    if type(num) == list:
        print([df.columns[i] for i in num])
    elif num:
        return list(enumerate(df.columns.tolist()))
    else:
        print(df.columns.tolist())

def check_se_type(se, type_):
    return all(se.map(lambda x: isinstance(x, type_)))

def show_valcount(se):
    df = se.value_counts().reset_index()
    df.columns = [se.name, '计数']
    showdf_in_em(df)

def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        #此处路径最好使用utf-8解码，否则在磁盘中可能会出现乱码的情况
        # os.makedirs(path.decode('utf-8')) 
        os.makedirs(path)
        return path
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return path

def html2csv(folder):
    csvfolder = mkdir(folder+'csv')
    for root,dirs,files in os.walk(folder, topdown=False):
        for name in tqdm(files):
            path = os.path.join(root,name)
            df = pd.read_html(path, encoding='gbk', header=0)[0]
            dfo.csv.save8(df, f'{csvfolder}\{name[:-4]}')

def showdf_in_em(df:pd.DataFrame,sep='\t'):
    # dataframe文本复制到剪贴板
    df.to_clipboard(sep=sep)
    dfo.operate_emeditor.ctrl_new()
    # 发送快捷键Ctrl+V和Alt+C+1
    shell.SendKeys("^v")  # 模拟按下 Ctrl+C
    shell.SendKeys(r"%c2")  # 模拟按下 Alt+C+1

def showdf_in_excel(df,sep='\t'):
    # dataframe文本复制到剪贴板
    df.to_clipboard(sep=sep)
    dfo.operate_excel.ctrl_new()
    # 发送快捷键Ctrl+V和Alt+C+1
    shell.SendKeys("^v")  # 模拟按下 Ctrl+C
#endregion

#%% 数据分析高级接口
def csv2dict(path, keycol='key', valcol='value', de_kna=True, de_vna=True):
    dctcsv = dfo.csv.read8(path)
    dctcsv = dctcsv.dropna(subset=[keycol]) if de_kna else dctcsv
    dctcsv = dctcsv.dropna(subset=[valcol]) if de_vna else dctcsv
    return dctcsv\
        .set_index([keycol])\
        [valcol]\
        .to_dict()

def df2dict(df, keycol='key', valcol='value', de_kna=True, de_vna=True):
    df = df.dropna(subset=[keycol]) if de_kna else df
    df = df.dropna(subset=[valcol]) if de_vna else df
    return df\
        .set_index([keycol])\
        [valcol]\
        .to_dict()

class MapDf(object):
    def __init__(self, df_or_path:'str|Path|pd.DataFrame') -> None:

        if isinstance(df_or_path, (str, Path)):
            self.read_csvdf(df_or_path)
        else:
            self.df = df_or_path
        
    @property
    def df(self): return self.__df
    
    @df.setter
    def df(self, df:pd.DataFrame): 
        assert isinstance(df, pd.DataFrame), \
            'What you passed in was not a data frame.'
        self.__df = df

    def read_csvdf(self, path:'str|Path'):
        self.df = dfo.csv.read8(path)

    # def __set_key(self, key:str) -> self:
    #     self.key = key
    #     return self
    
    # def __set_val(self, val:str) -> self:
    #     self.val = val
    #     return self
    
    # def setkv(self, key:str, val:str) -> self:
    #     self.key = key
    #     self.val = val
    #     return self


    def todict(self, key:str, val:str) -> dict: 
        keyval_undup_df = self.df.loc[:,[key,val]]\
            .drop_duplicates([key,val])
        keydup_index = keyval_undup_df.duplicated([key], keep=False)
        if any(keydup_index):
            display(keyval_undup_df[keydup_index])
            raise KeyError(
                'Multiple values for the same key, '
                'these keys and their corresponding values are shown above, '
                'please correct errors in the rule dictionary used for mapping.')
        return keyval_undup_df\
            .set_index([key])\
            [val]\
            .to_dict()

def checkmap(self:pd.Series, dct:dict) -> pd.Series:
    new_se = self.map(dct)
    fail2map_data = self.loc[new_se.isna()].unique()
    if len(fail2map_data)>0:
        print('以下内容匹配失败，请检查：',
            '\n',
            fail2map_data.tolist(),
            '\n')
    return new_se
