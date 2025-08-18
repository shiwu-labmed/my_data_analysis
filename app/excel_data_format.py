
#%% 导入包
import pandas as pd 
import numpy as np 
import tqdm
import os
import re
from IPython.display import display

import win32con
import win32gui
import win32com.client
from pathlib import *
Path.cwd()

from ..base_l1 import datafile_operate as dfo 

shell = win32com.client.Dispatch("WScript.Shell")


#%%
def excel_date_format(s:str, format:str='yyyy年mm月')->str:
    return f'=TEXT({s},"{format}")'

def excel_yyyymm(s:str)->str:
    return f'=TEXT({s},"yyyy年mm月")'

def excel_yyyym(s:str)->str:
    return f'=TEXT({s},"yyyy年m月")'