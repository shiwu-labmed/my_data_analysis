#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' 辅助快速数据分析的函数 '

__author__ = 'ShiLiu'

#%% 导入包
#系统包
from pathlib import Path

#数据分析包
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects

#其他包
import pysnooper
# from IPython.display import display

#自编包
from ..base_l2 import Simplified_pandas as spd
from ..base_l1 import datafile_operate as dfo 
dfo.prepare_path()

#%% 多组p值对比线
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

if __name__=='__main__':
    pass

#%% 堆叠柱状图
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

#%% 小提琴图绘图类
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
    
#%% mpl文本描边工具
class mpl_txt_stroke(object):
    def __init__(self, linewidth=3, foreground='k', **other_stroke_style) -> None:
        self.stroke_style = PathEffects.withStroke(
            linewidth=linewidth, foreground=foreground, **other_stroke_style)

    def stroke(self, txt_obj):
        txt_obj.set_path_effects([self.stroke_style])

    def stroke_multitxt(self, txt_obj_iterator):
        for txt_obj in txt_obj_iterator:
            txt_obj.set_path_effects([self.stroke_style])

#%% 画多个roc
@pysnooper.snoop('./mul_roc类.log')
class multi_roc(object):
    
    def __init__(self, path='多个AUC数据') -> None:
        self.aucdata_path = path
        # if os.path.exists(self.aucdata_path):
        if Path(self.aucdata_path).exists():
            self.load_aucdata()
        else:
            self.aucdata = pd.DataFrame(
                columns=['fpr','tpr','thresholds','auc'])
            self.save_aucdata()

    @property
    def aucdata_path(self):
        return self._aucdata_path

    @aucdata_path.setter
    def aucdata_path(self, path:str):
        path = Path(dfo.adjust_path(path, 'csv'))
        if '多个AUC数据' not in path.stem:
            path = path.parent/f'{path.stem}_多个AUC数据{path.suffix}'
        self._aucdata_path = path
    
    def load_aucdata(self):
        self.aucdata = dfo.csv.read8(self.aucdata_path, index_col=0)
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
        spd.col2num(self.aucdata, ['auc'])
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
        dfo.csv.save8(df, self.aucdata_path, index=True)
    
    @property
    def ax(self):
        return self._ax

    def ax(self, ax):
        self._ax = ax
    
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

#%% roc快速绘图函数
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

#%% 设置mpl字体
def mpl_font_zh(font_name:str=None):
    winfont_path = Path('C:\Windows\Fonts')
    if font_name is None:
        [print(str(x)) for x in winfont_path.glob("*")]
        raise ValueError(
            "The installed fonts in your windows system "\
            "are already printed above, please select one "\
            "and pass its name into the fontname parameter")
    # if all([i not in font_path for i in ['\\','/']]):
    #     font_path = f'D:\python_related\MatplotlibFont\{font_path}'
    else:
        font_path = winfont_path.regex1path(font_name)
        fe = mpl.font_manager.FontEntry(
            fname=font_path,
            name=font_path.name)
        mpl.font_manager.fontManager.ttflist.append(fe)
        mpl.rcParams['font.sans-serif']= [fe.name]
        print(font_path, ' has been set as the default font for mpl')

def mpl_use_GlowSans_compress_regular():
    mpl_font_zh('GlowSansSC-Compressed-Regular.otf')

if __name__=='__main__':
    mpl_font_zh()
    mpl_font_zh('Glow.*otf')

#%% 分割图设置
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

