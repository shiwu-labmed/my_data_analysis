#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' 辅助快速数据分析的函数 '

__author__ = 'ShiLiu'

#%% 导入依赖包
import pysnooper
import sys
from turtle import position
import win32com.client
import win32gui
import win32con
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit #分层抽样
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import os
from tqdm import tqdm
import IPython
import time

shell = win32com.client.Dispatch("WScript.Shell")

#%% pandas函数和类
#region dataframe读取写入
def read_csv(file_name, my_dtype=object, encoding='gbk', **kw):
    if '.csv' not in file_name:
        file_name = '%s.csv'%file_name
    if all([i not in file_name for i in ['\\','/']]):
        file_name = r'.\%s'%file_name
    return pd.read_csv(filepath_or_buffer=file_name, encoding=encoding, dtype=my_dtype, keep_default_na=False, na_values=['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A','#N/A', 'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''], **kw)

def read_csv8(file_name, my_dtype=object, **kw):
    if '.csv' not in file_name:
        file_name = '%s.csv'%file_name
    if all([i not in file_name for i in ['\\','/']]):
        file_name = r'.\%s'%file_name
    return pd.read_csv(filepath_or_buffer=file_name, encoding='utf-8', dtype=my_dtype, keep_default_na=False, na_values=['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A','#N/A', 'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', ''], **kw)

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

def rand_df():
    return pd.DataFrame(np.random.rand(3,4), columns=list('ABCD'))

#endregion

#region dataframe操作
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
            save_csv8(df, f'{csvfolder}\{name[:-4]}')

def showdf_in_em(df:pd.DataFrame,sep='\t'):
    # dataframe文本复制到剪贴板
    df.to_clipboard(sep=sep)
    operate_emeditor.ctrl_new()
    # 发送快捷键Ctrl+V和Alt+C+1
    shell.SendKeys("^v")  # 模拟按下 Ctrl+C
    shell.SendKeys(r"%c2")  # 模拟按下 Alt+C+1

def showdf_in_excel(df,sep='\t'):
    # dataframe文本复制到剪贴板
    df.to_clipboard(sep=sep)
    operate_excel.ctrl_new()
    # 发送快捷键Ctrl+V和Alt+C+1
    shell.SendKeys("^v")  # 模拟按下 Ctrl+C
#endregion

#%% 数据分析高级接口
def csv2dict(path, keycol='key', valcol='value', de_kna=True, de_vna=True):
    dctcsv = read_csv8(path)
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
    def __init__(self) -> None:
        pass
    
    @property
    def df(self): return self.__df
    
    @df.setter
    def df(self, df): self.__df = df

    def read_df(self, path):
        self.df = read_csv8(path)

    def dup_row(self, col):
        dup_bool = self.df[col].duplicated(keep=False)
        return self.df[col].reset_index()[dup_bool]
    
    def check_dup(self, col):
        dup_row = self.dup_row()
        if len(dup_row)>0:
            IPython.display.display(dup_row)
        

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

# os.path.splitext
# bloodcul_MLfeature = MLfeature(
#     'D:\\作业文件\\研究生\\研究生课题\\机器学习血流感染\\基本数据\\'\
#         '变量表原始数据\\血培养机器学习_特征vs重要性.csv')   

#%% 系统操作函数
def file_in_folder(folder_path, file_type='.csv'):
    file_path_list = []
    for root,dirs,files in os.walk(folder_path, topdown=False):
        for name in files:
            if os.path.splitext(name)[1] == file_type:
                file_path_list.append(os.path.join(root,name))
    return file_path_list

class get_file_from(object):

    def __init__(self,source_floder) -> None:
        self.source_floder = source_floder
        self.basename = maplist(os.listdir(source_floder))
        self.abspath = self.basename\
            .map(lambda x: f'{self.source_floder}\\{x}')

    def sufx(self, suffix):
        self.abspath = self.abspath\
            .filter(lambda x: os.path.splitext(x)[1]==suffix)
        return self

    def base_name(self):
        return self.abspath.map(os.path.basename).list
    
    def base_name_nosuf(self):
        return self.abspath\
                .map(os.path.basename)\
                .map(lambda x: os.path.splitext(x)[0])\
                .list
    
    def adjust_path_list(self, basename=False, no_suffix=False):
        adjust_all = not any([basename, no_suffix])
        if basename or adjust_all:
            self.basename=self.base_name()
        if no_suffix or adjust_all:
            self.basename_nosuf=self.base_name_nosuf
        return self

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

#%% 绘图函数
def mpl_font_zh(font_path):
    if all([i not in font_path for i in ['\\','/']]):
        font_path = f'D:\python_related\MatplotlibFont\{font_path}'
    fe = mpl.font_manager.FontEntry(
        fname=font_path,
        name=os.path.basename(font_path))
    mpl.font_manager.fontManager.ttflist.append(fe)
    mpl.rcParams['font.sans-serif']= [fe.name]

# 分割图设置
def set_spines_cutLine(ax_above, ax_below, **cutline_style):
    # 隐藏上方坐标轴的底边，隐藏下方坐标轴的顶边
    ax_above.spines.bottom.set_visible(False)
    ax_below.spines.top.set_visible(False)
    ax_above.xaxis.tick_top()
    # ax_above.axes.xaxis.set_visible(False)
    ax_above.tick_params(top=False)
    ax_above.tick_params(labeltop=False)  # don't put tick labels at the top
    ax_below.xaxis.tick_bottom()

    d = 0  # proportion of vertical to horizontal extent of the slanted line
    inner_cutline_style = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    inner_cutline_style.update(cutline_style)
    ax_above.plot(
        [0, 1], [0, 0], transform=ax_above.transAxes, **inner_cutline_style)
    ax_below.plot(
        [0, 1], [1, 1], transform=ax_below.transAxes, **inner_cutline_style)

def plot_roc(fpr, tpr, label):
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

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

#%% 绘图类
class multi_g_pval(object):
    
    def __init__(self, ax, groupxs) -> None:
        self.ax = ax
        self.groupxs = groupxs
        xmin,xmax = ax.get_xlim()
        xrange = xmax-xmin
        self.groups_x_axes_frac = [(x-xmin)/xrange for x in self.groupxs]
        self.pval_plot_pattern = dict(
            ha='center', va='center',
            xycoords='axes fraction', 
            arrowprops=dict(
                arrowstyle='|-|, widthA=0, widthB=0.2', 
                lw=1, color='black', shrinkA=0.3, shrinkB=0))

    def plot_pval(self,gleft,gright,y,pval_sign,**pval_style):
        gleft_x = self.groups_x_axes_frac[gleft-1]
        gright_x = self.groups_x_axes_frac[gright-1]
        star_x = np.mean([gleft_x,gright_x])
        self.pval_plot_pattern.update(pval_style)
        self.ax.annotate(pval_sign, xy=(gleft_x, y), xytext=(star_x,y), **self.pval_plot_pattern)
        self.ax.annotate(pval_sign, xy=(gright_x, y), xytext=(star_x,y), **self.pval_plot_pattern)
        return self

class stack_bar(object):
    def __init__(self,ax) -> None:
        self.ax = ax
        self.heights = []
        self.bars = []
        self.bar_style = {}
        self.label_style = {}
        self.height_labels = {}
    
    def bar(self,height,x=None,**bar_style_kw):
        self.group_ticks = x if x is not None else range(len(height))
        self.bar_style = bar_style_kw
        bar = self.ax.bar(self.group_ticks, height, **self.bar_style)
        self.heights.append(height)
        self.bars.append(bar)
        self.heights_sum = height
    
    def stack(self, height, xory=None, **bar_style_kw):
        self.group_ticks = xory if xory is not None else self.group_ticks
        self.bar_style.update(bar_style_kw)
        stack_bar = self.ax.bar(self.group_ticks, height, bottom=self.heights_sum, **self.bar_style)
        self.heights.append(height)
        self.bars.append(stack_bar)
        self.heights_sum += height

    def label_heights(self, stack_num='all', hide_zero=True, **txt_style):
        self.label_style.update(txt_style)
        stack_num = list(range(len(self.bars))) if stack_num == 'all' else stack_num
        stack_num = [stack_num] if type(stack_num)==int else stack_num
        for num in stack_num:
            bar,height = self.bars[num],self.heights[num]
            label_txts = self.ax.bar_label(bar, height, **self.label_style)
            if hide_zero:
                for txt in label_txts:
                    if txt.get_text()=='0':
                        txt.set_visible(False) 
            self.height_labels[num] = label_txts

    def label_heights_sum(self, hide_zero=True, **txt_style):
        self.label_style.update(txt_style)
        bar,height = self.bars[-1],self.heights_sum
        label_txts = self.ax.bar_label(bar, height, **self.label_style)
        if hide_zero:
            for txt in label_txts:
                if txt.get_text()=='0':
                    txt.set_visible(False)
        self.heights_sum_label = label_txts

class stack_barh(object):
    def __init__(self,ax) -> None:
        self.ax = ax
        self.widths = []
        self.barhs = []
        self.barh_style = {}
        self.label_style = {}
        self.width_labels = {}

    def barh(self,width,x=None,**barh_style_kw):
        self.group_ticks = x if x is not None else range(len(width))
        self.barh_style = barh_style_kw
        barh = self.ax.barh(self.group_ticks, width, **self.barh_style)
        self.widths.append(width)
        self.barhs.append(barh)
        self.widths_sum = width.copy()
    
    def stack(self, width, xory=None, **barh_style_kw):
        self.group_ticks = xory if xory is not None else self.group_ticks
        self.barh_style.update(barh_style_kw)
        stack_barh = self.ax.barh(
            self.group_ticks, width, left=self.widths_sum, **self.barh_style)
        self.widths.append(width)
        self.barhs.append(stack_barh)
        self.widths_sum += width
        return None

    def label_widths(self, stack_num='all', hide_zero=True, **txt_style):
        self.label_style.update(txt_style)
        stack_num = list(range(len(self.barhs))) if stack_num == 'all' else stack_num
        stack_num = [stack_num] if type(stack_num)==int else stack_num
        for num in stack_num:
            barh,width = self.barhs[num],self.widths[num]
            label_txts = self.ax.bar_label(barh, width, **self.label_style)
            if hide_zero:
                for txt in label_txts:
                    if txt.get_text()=='0':
                        txt.set_visible(False)
            self.width_labels[num] = label_txts

    def label_widths_sum(self, hide_zero=True, **txt_style):
        self.label_style.update(txt_style)
        barh,width = self.barhs[-1],self.widths_sum
        widths_sum_label_txt = self.ax.bar_label(barh, width, **self.label_style)
        if hide_zero:
            for txt in widths_sum_label_txt:
                if txt.get_text()=='0':
                    txt.set_visible(False)
        self.widths_sum_label = widths_sum_label_txt

# 小提琴图绘图类
class violin_data(object):
    def __init__(self, data):
        data = data.dropna()
        self.data = data
        self.datasize = len(data)
        self.q1, self.median, self.q3 = np.percentile(data,[25, 50, 75], axis=0)
        self.iqr = self.q3 - self.q1
        self.whisker_max = self.q3 + 1.5*self.iqr
        self.whisker_max = np.clip(self.whisker_max, self.q3, data.max())
        self.whisker_min = self.q1 - 1.5*self.iqr
        self.whisker_min = np.clip(self.whisker_min, data.min(), self.q1)
        
        # 创建默认参数
        self.vioplot_args = dict(widths=0.7)
        self.vio_style = dict(
            facecolor='#C8C8C8', edgecolor='black', alpha=1, zorder=2)
        self.median_style = dict(marker='o', color='white', s=24, zorder=3)
        self.iqr_style = dict(color='k', linestyle='-', lw=4)
        self.whisker_style = dict(color='k', linestyle='-', lw=1)

    def vioplot(
            self,ax,x,
            vioplot_args=dict(), vio_style=dict(), median_style=dict(),
            iqr_style=dict(), whisker_style=dict()):
        # 更新样式
        self.vioplot_args.update(vioplot_args)
        self.vio_style.update(vio_style)
        self.median_style.update(median_style)
        self.iqr_style.update(iqr_style)
        self.whisker_style.update(whisker_style)

        # 画小提琴图
        self.ax =  ax
        parts = self.ax.violinplot(
            self.data, [x], 
            showmeans=False, showmedians=False,showextrema=False,
            **vioplot_args)
        
        # 调整小提琴形状样式
        self.pc = parts['bodies'][0]
        self.pc.set(**self.vio_style)

        # 画内部箱线图
        self.ax.scatter(x, self.median, **self.median_style)
        self.ax.vlines(x, self.q1, self.q3, **self.iqr_style)
        self.ax.vlines(x, self.whisker_min, self.whisker_max, **self.whisker_style)

class violin_plot(object):
    def __init__(self,ax) -> None:
        self.ax = ax
        # 创建默认样式
        self.vioplot_args = dict(widths=0.7)
        self.vio_style = dict(
            facecolor='#C8C8C8', edgecolor='black', alpha=1, zorder=2)
        self.median_style = dict(marker='o', color='white', s=24, zorder=3)
        self.iqr_style = dict(color='k', linestyle='-', lw=4)
        self.whisker_style = dict(color='k', linestyle='-', lw=1)
        # 创建数据容器
        self.violins = dict()

    def vioplot(
            self,data,x,
            vioplot_args=dict(), vio_style=dict(), median_style=dict(),
            iqr_style=dict(), whisker_style=dict()):
        # 更新样式
        self.vioplot_args.update(vioplot_args)
        self.vio_style.update(vio_style)
        self.median_style.update(median_style)
        self.iqr_style.update(iqr_style)
        self.whisker_style.update(whisker_style)

        # 画小提琴图
        violin = violin_data(data)
        violin.vioplot(
            self.ax, x, 
            self.vioplot_args, self.vio_style, self.median_style, 
            self.iqr_style, self.whisker_style)
        self.violins[x] = violin

    def add_datasize(self, str_form=' (n={0})'):
        xlabels = [label.get_text() for label in self.ax.get_xticklabels()]
        self.x_s = list(self.violins.keys())
        self.x_s.sort()
        datasizes = [self.violins[x].datasize for x in self.x_s]
        new_xlabels = [xl+str_form.format(ds) for xl,ds in zip(xlabels, datasizes)]
        self.ax.set_xticks(self.x_s, new_xlabels)

    def get_violin(self,num):
        self.x_s = list(self.violins.keys())
        self.x_s.sort()
        return self.violins[self.x_s[num-1]]

    def all_violin_data(self):
        self.violins_describe=pd.DataFrame()
        self.x_s = list(self.violins.keys())
        self.x_s.sort()
        for x in self.x_s:
            violin = self.violins[x]
            violin_df = pd.DataFrame({
                '横坐标':[x],
                '数据量':[violin.datasize],
                '须的范围':[f'{violin.whisker_min:.2f}-{violin.whisker_max:.2f}'],
                'IQR':[f'{violin.q1:.2f}-{violin.q3:.2f}'],
                '中位数':[f'{violin.median:.2f}']})
            self.violins_describe = pd.concat(
                [self.violins_describe,violin_df],
                ignore_index=True)
        return self.violins_describe

class mpl_txt_stroke(object):
    def __init__(self, linewidth=3, foreground='k', **other_stroke_style) -> None:
        self.stroke_style = PathEffects.withStroke(
            linewidth=linewidth, foreground=foreground, **other_stroke_style)

    def stroke(self, txt_obj):
        txt_obj.set_path_effects([self.stroke_style])

    def stroke_multitxt(self, txt_obj_iterator):
        for txt_obj in txt_obj_iterator:
            txt_obj.set_path_effects([self.stroke_style])


@pysnooper.snoop('./mul_roc类.log')
class multi_roc(object):
    
    def __init__(self, path='多个AUC数据') -> None:
        self.set_aucdata_path(path)
        if os.path.exists(self.aucdata_path):
            self.load_aucdata()
        else:
            self.aucdata = pd.DataFrame(
                columns=['fpr','tpr','thresholds','auc'])
            self.save_aucdata()
    
    def set_aucdata_path(self, path='多个AUC数据'):
        if '.csv' not in path:
            path = '%s.csv'%path
        if all([i not in path for i in ['\\','/']]):
            path = r'.\%s'%path
        self.aucdata_path = path
    
    def load_aucdata(self):
        self.aucdata = read_csv8(self.aucdata_path, index_col=0)
        self.adj_col4calculate()

    def cell2array(self, cell):
        if type(cell)==str:
            cell = eval(cell)
        if type(cell)==list:
            cell = np.array(cell)
        return cell
        
    def cell2listr(self, cell):
        if type(cell) in [np.array, np.ndarray]:
            cell = str(list(cell))
        return cell
        
    def adj_col4calculate(self):
        col2num(self.aucdata, ['auc'])
        for col in ['fpr', 'tpr', 'thresholds']:
            self.aucdata[col] = self.aucdata[col]\
                .map(self.cell2array)
            
    def adj_col4save(self, df):
        for col in ['fpr', 'tpr', 'thresholds']:
            df[col] = df[col].map(self.cell2listr)
        return df

    def save_aucdata(self):
        df = self.aucdata.copy()
        df = self.adj_col4save(df)
        save_csv8(df, self.aucdata_path, index=True)
    
    def set_ax(self, ax):
        self.ax = ax
    
    def add(self, modelname, fpr, tpr, thresholds=None, auc=np.nan):
        self.aucdata.loc[modelname, 'fpr'] = str(list(fpr))
        self.aucdata.loc[modelname, 'tpr'] = str(list(tpr))
        self.aucdata.loc[modelname, 'thresholds'] = \
            str(list(
                thresholds 
                if thresholds is not None 
                else np.array([0]*len(tpr))
                ))
        self.aucdata.loc[modelname, 'auc'] = auc
        self.adj_col4calculate()
        self.save_aucdata()

    def __xy(self, modelname):
        return [
            self.aucdata.loc[modelname,'fpr'], 
            self.aucdata.loc[modelname,'tpr']
            ]
    
    def __label(self, modelname, model_label):
        auc_val = self.aucdata.loc[modelname,"auc"]
        return f'{model_label} (AUC = {auc_val:.2f})'
    
    def plot(self, modelname, model_label, **kwargs):
        self.ax.plot(
            *self.__xy(modelname), 
            label = self.__label(modelname, model_label),
            **kwargs)
    
    def cal_opt_senspetsh(self):
        self.aucdata['opt_youden_pos'] = self.aucdata\
            .apply(lambda se: np.argmax(se['tpr']-se['fpr']), axis=1)
        self.aucdata['opt_sen'] = self.aucdata\
            .apply(lambda se: se['tpr'][se['opt_youden_pos']], axis=1)
        self.aucdata['opt_spe'] = self.aucdata\
            .apply(lambda se: 1-se['fpr'][se['opt_youden_pos']], axis=1)
        self.aucdata['opt_threshold'] = self.aucdata\
            .apply(lambda se: se['thresholds'][se['opt_youden_pos']], axis=1)
        self.save_aucdata()

    def opt_data(self):
        self.cal_opt_senspetsh()
        return self.aucdata\
            .loc[:,['opt_youden_pos', 'opt_sen', 'opt_spe', 'opt_threshold']]

#%% 测试代码
if __name__ == '__main__':
    html2csv(r'D:\作业文件\研究生\【1】文章\hiv总\实验室检查原始数据\17-22实验室检查')
    html2csv(r'D:\作业文件\研究生\【1】文章\hiv总\实验室检查原始数据\测试')
