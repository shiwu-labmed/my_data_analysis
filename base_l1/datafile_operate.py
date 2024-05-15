#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' 操作excel、csv、剪贴板表格，包括读写，修改 '

__author__ = 'ShiLiu'

#%% 导入依赖包
import pysnooper
import win32con
import win32gui
import win32com.client
import time
import os

import pandas as pd
import numpy as np
import pyperclip
import re
from IPython.display import display
from pathlib import *
Path.cwd()

shell = win32com.client.Dispatch("WScript.Shell")

#%% 调整路径字符串
def adjust_path(path:str, suffix=''):
    if suffix not in path:
        path = f'{path}.{suffix}'
    if all([i not in path for i in ['\\','/']]):
        path = f'./{path}'
    return path

#%% 路径操作
#通配符匹配当前路径下的文件
def glob1(self:Path, path_regex:str) -> Path:
    matched_path = [path for path in self.glob(path_regex)]
    matched_path_num = len(matched_path)
    if matched_path_num==1:
        return matched_path[0]
    else:
        print(
            'ERROR INFO: \n'\
            f'the regex "{path_regex}" matched {matched_path_num} path under the following path: \n'\
            f'{str(self)}, \n'\
            'please modify the regex to ensure that it matches only one path, \n'\
            'The current matching paths are as follows: \n',
            '\n'.join([f'...\{x.name}' for x in matched_path]))
        raise KeyError('error info has been printed above')

if __name__=='__main__':
    Path.glob1 = glob1
    Path.cwd().glob1('*data*')
    Path.cwd().glob1('.*data.*(ana|oper).*')
    [x for x in Path.cwd().rglob('.*data.*(ana|oper).*')]

#%% 正则表达式匹配当前路径下的文件
def regex_path(self:Path, path_regex:str) -> list:
    path_regex_ = re.compile(path_regex)
    matched_paths_name = [path.name for path in self.iterdir() if path_regex_.match(path.name)]
    matched_paths = [self/pathname for pathname in matched_paths_name]
    return matched_paths

if __name__=='__main__':
    [x for x in Path.cwd().rglob('.*data.*(ana|oper).*')]
    Path.regex_path = regex_path
    Path.cwd().regex_path('.*data.*(ana|oper).*')
    # re.compile('.*data.*(ana|oper).*').match('datafile_operate.py')
#%% 正则表达式匹配当前路径下的文件_限1个
def regex1path(self:Path, path_regex:str) -> Path:
    matched_paths = self.regex_path(path_regex)
    matched_path_num = len(matched_paths)
    if matched_path_num==1:
        return matched_paths[0]
    else:
        print(
            'ERROR INFO: \n'\
            f'the regex "{path_regex}" matched {matched_path_num} path under the following path: \n'\
            f'{str(self)}, \n'\
            'please modify the regex to ensure that it matches only one path, \n'\
            'The current matching paths are as follows: \n',
            '\n'.join([f'...\{x.name}' for x in matched_paths]))
        raise KeyError('error info has been printed above')

if __name__=='__main__':
    [x for x in Path.cwd().rglob('.*data.*(ana|oper).*')]
    Path.regex_path = regex_path
    Path.regex1path = regex1path
    Path.cwd().regex1path('.*data.*(ana|oper).*')
    # re.compile('.*data.*(ana|oper).*').match('datafile_operate.py')

#%%
def prepare_path():
    global Path
    Path.glob1 = glob1
    Path.regex_path = regex_path
    Path.regex1path = regex1path

#%% 辅助编码的代码
def set_misspara_as_default(default, *args):
    return [x if bool(x) else default for x in args]

if __name__=='__main__':
    set_misspara_as_default(1, 3,'r',False)

#%% excel操作类
class Excel(object):
    def __init__(self) -> None:
        pass

    def adjust_csv_path(self, path):
        path = str(path) if isinstance(path, Path) else path
        assert isinstance(path, str), \
            'you can only pass in a path string or a Path object'
        if '.xlsx' not in path:
            path = f'{path}.xlsx'
        if all([i not in path for i in ['\\','/']]):
            path = str(Path.cwd()/path)
        return path
    
    def readgb(self, file_name, sheet_name=0, dtype=object, encoding='gbk', **kw):
        file_name = self.adjust_csv_path(file_name)
        return pd.read_excel(
            io=file_name,
            sheet_name=sheet_name,
            encoding=encoding, 
            dtype=dtype, 
            keep_default_na=False, 
            na_values=[
                '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A','#N/A', 
                'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''], 
            **kw)
    
    def read8(self, file_name, sheet_name=0, dtype=object, encoding='utf-8', **kw):
        return self.readgb(file_name, sheet_name, dtype, encoding, **kw)
    
    def savegb(self):
        pass

excel = Excel()

if __name__=='__main__':
    pass

#%% csv操作类
class Csv(object):
    def __init__(self) -> None:
        pass

    def adjust_csv_path(self, path):
        path = str(path) if isinstance(path, Path) else path
        assert isinstance(path, str), \
            'you can only pass in a path string or a Path object'
        if '.csv' not in path:
            path = f'{path}.csv'
        if all([i not in path for i in ['\\','/']]):
            path = str(Path.cwd()/path)
        return path

    def read(self, file_name, dtype=object, encoding='gbk', sep=',', **kw) -> pd.DataFrame:
        file_name = self.adjust_csv_path(file_name)
        return pd.read_csv(
            filepath_or_buffer=file_name, 
            encoding=encoding, 
            dtype=dtype, 
            sep=sep,
            keep_default_na=False, 
            na_values=[
                '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A','#N/A', 
                'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''], 
            **kw)

    def read8(self, file_name, dtype=object, sep=',', **kw) -> pd.DataFrame:
        return self.read(file_name, dtype=dtype, encoding='utf-8', sep=sep, **kw)

    def save(self, df:pd.DataFrame, path, index=False, encoding='gbk',**kw):
        path = self.adjust_csv_path(path)
        df.to_csv(path, index=index, encoding=encoding, **kw)

    def save8(self, df:pd.DataFrame, path, index=False, encoding='utf-8', **kw):
        self.save(df, path, index=index, encoding=encoding, **kw)

    def trans(
            self, 
            file_name:'Path|str',
            output:'Path|str' = 'clipboard',
            sep_default:str=',', sepfrom:'str|False'=False, septo:'str|False'=False, 
            ec_default:'str'='utf-8', ecfrom:'str|False'=False, ecto:'str|False'=False, 
            ) -> str: 
        sepfrom,septo = [x if bool(x) else sep_default for x in [sepfrom,septo]]
        ecfrom,ecto = [x if bool(x) else ec_default for x in [ecfrom,ecto]]
        df = self.read(file_name, dtype=object, encoding=ecfrom, sep=sepfrom) 
        if output=='clipboard':
            df.to_clipboard(sep=septo)
        else:
            path = self.adjust_csv_path(output)
            self.save(df, path, index=False, encoding=ecto, sep=septo)

csv = Csv()

if __name__=='__main__':
    pass

#%% 剪贴板操作类
class Clipboard(object):
    def __init__(self) -> None:
        pass

    def read(self, dtype=object, encoding='utf-8', sep=',', **kw) -> pd.DataFrame:
        return pd.read_clipboard(
            encoding=encoding, 
            dtype=dtype, 
            sep=sep,
            keep_default_na=False, 
            na_values=[
                '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A','#N/A', 
                'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''], 
            **kw)
    
    def read8(self, dtype=object, sep=',', **kw) -> pd.DataFrame:
        return self.read(dtype=object, encoding='utf-8', sep=sep, **kw)
    
    def trans(
            self, 
            sep_default:str=',', sepfrom:'str|False'=False, septo:'str|False'=False, 
            ec_default:'str'='utf-8', ecfrom:'str|False'=False, ecto:'str|False'=False, 
            output:'Path|str'='clipboard', outputindex=False):
        sepfrom,septo = [x if bool(x) else sep_default for x in [sepfrom,septo]]
        ecfrom,ecto = [x if bool(x) else ec_default for x in [ecfrom,ecto]]
        df = self.read(dtype=object, encoding='utf-8', sep=',')
        if output=='clipboard':
            df.to_clipboard(index=outputindex, sep=septo)
        else:
            csv.save(df, output, index=outputindex, encoding=ecto, sep=septo)

clipb = Clipboard()

if __name__=='__main__':
    clipb.trans(septo='\t')


#%% 旧版函数读写操作，已弃用
#region dataframe读取写入
def read_csv(file_name, my_dtype=object, encoding='gbk', **kw):
    if '.csv' not in file_name:
        file_name = '%s.csv'%file_name
    if all([i not in file_name for i in ['\\','/']]):
        file_name = r'.\%s'%file_name
    return pd.read_csv(
        filepath_or_buffer=file_name, 
        encoding=encoding, 
        dtype=my_dtype, 
        keep_default_na=False, 
        na_values=[
            '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A',
            '#N/A', 'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''], 
        **kw)

def read_csv8(file_name, my_dtype=object, **kw):
    if '.csv' not in file_name:
        file_name = '%s.csv'%file_name
    if all([i not in file_name for i in ['\\','/']]):
        file_name = r'.\%s'%file_name
    return pd.read_csv(
        filepath_or_buffer=file_name, 
        encoding='utf-8', 
        dtype=my_dtype, 
        keep_default_na=False, 
        na_values=[
            '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A',
            '#N/A', 'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''], 
        **kw)

def save_csv(df, path, index=False, **kw):
    if '.csv' not in path:
        path = '%s.csv'%path
    if all([i not in path for i in ['\\','/']]):
        path = r'.\%s'%path
    df.to_csv(path, index=index, encoding='gbk', **kw)

def save_csv8(df, path, index=False, **kw):
    if '.csv' not in path:
        path = '%s.csv'%path
    if all([i not in path for i in ['\\','/']]):
        path = r'.\%s'%path
    df.to_csv(path, index=index, encoding='utf-8', **kw)

#endregion

#%%
# def file_in_folder(folder_path, file_type='.csv'):
#     file_path_list = []
#     for root,dirs,files in os.walk(folder_path, topdown=False):
#         for name in files:
#             if os.path.splitext(name)[1] == file_type:
#                 file_path_list.append(os.path.join(root,name))
#     return file_path_list

# class get_file_from(object):

#     def __init__(self,source_floder) -> None:
#         self.source_floder = source_floder
#         self.basename = maplist(os.listdir(source_floder))
#         self.abspath = self.basename\
#             .map(lambda x: f'{self.source_floder}\\{x}')

#     def sufx(self, suffix):
#         self.abspath = self.abspath\
#             .filter(lambda x: os.path.splitext(x)[1]==suffix)
#         return self

#     def base_name(self):
#         return self.abspath.map(os.path.basename).list
    
#     def base_name_nosuf(self):
#         return self.abspath\
#                 .map(os.path.basename)\
#                 .map(lambda x: os.path.splitext(x)[0])\
#                 .list
    
#     def adjust_path_list(self, basename=False, no_suffix=False):
#         adjust_all = not any([basename, no_suffix])
#         if basename or adjust_all:
#             self.basename=self.base_name()
#         if no_suffix or adjust_all:
#             self.basename_nosuf=self.base_name_nosuf
#         return self

def winlnk(inkpath):
    if '.lnk' not in inkpath:
        inkpath = f'{inkpath}.lnk'
    if all([i not in inkpath for i in ['\\','/']]):
        inkpath = f'./{inkpath}'
    shortcut = shell.CreateShortCut(inkpath)
    return shortcut.Targetpath

# 这个感觉还配不上这个名字hh
class operate_software(object):
    def __init__(self, software_path, software_window_class) -> None:
        self.software_path = software_path
        self.software_window_class = software_window_class
    
    def set_win_fore(self, hwnd):
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)

    def wait_win_fore(self):
        foreground_window_classname = win32gui.GetClassName(
            win32gui.GetForegroundWindow())
        while foreground_window_classname != self.software_window_class:
            time.sleep(0.1)
            foreground_window_classname = win32gui.GetClassName(
                win32gui.GetForegroundWindow())
        win32gui.ShowWindow(
            win32gui.GetForegroundWindow(), 
            win32con.SW_SHOWMAXIMIZED)

    def ctrl_new(self):
        hwnd = win32gui.FindWindow(self.software_window_class, None)
        if hwnd==0:
            os.startfile(self.software_path)
        else:
            self.set_win_fore(hwnd)
        self.wait_win_fore()
        time.sleep(0.1)
        shell.SendKeys("^n")

operate_excel = operate_software(
    'C:\\Program Files (x86)\\Microsoft Office\\root'\
        '\\Office16\\EXCEL.EXE',
    'XLMAIN')

operate_emeditor = operate_software(
    r'D:\professional software\EmEditor\EmEditor\EmEditor.exe',
    'EmEditorMainFrame3')
