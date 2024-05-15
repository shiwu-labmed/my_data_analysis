#%%
import pandas as pd 
import numpy as np 
from IPython.display import display

#%%
def series_valcout_and_proporation(
        se:pd.Series, countcol_name=['unique_values']) -> pd.DataFrame:
    valcount = se.value_counts()
    proporation = se.value_counts(normalize=True)
    valcount_and_proporation = pd.concat(
        [valcount, proporation], axis=1)
    valcount_and_proporation.index.names = countcol_name
    # display(valcount_and_proporation)
    return valcount_and_proporation


if __name__=='__main__':
    df = pd.DataFrame({
        '000':[1,1,1,1,2,2,2,2],
        '333':[3,3,3,2,2,2,1,1],
        '111':[1,2,1,2,3,2,1,2],
        '222':[1,2,1,2,2,2,1,1],
         })
    pd.Series([1,2,1,2,2],)
    series_valcout_and_proporation(df['111'])
    df.groupby('000')['111'].apply(series_valcout_and_proporation)
    df.groupby('333')['111'].apply(series_valcout_and_proporation).reset_index()
    df.groupby('333')['111'].apply(lambda se: print(se.describe()))
    df.groupby('333')['111'].apply(lambda se: se.name)
    df.groupby('333')['111'].apply(lambda se: se.value_counts()).reindex()

