# ----------------------------------------------------------------------
#                          Share turnover
# 
# Share turnover is average of daily turnover during past n-month
# Daily turnover is traded volumes divided by total number of shares
# By default, there is no adjustment for NASDAQ volumes
# If you need adjustment,  please set `adjustment` to True
# The adjustment is based on Gao and Ritter (2010)
# ----------------------------------------------------------------------

import pandas as pd
import numpy as np


class ap_share_turnover:
    def __init__(self, fpath):
        vars_list = ['permno', 'date', 'exchcd', 'shrcd', 'shrout', 'vol']
        _df1 = pd.read_parquet(fpath+'dsf1.parquet.gzip',
            columns=vars_list)
        _df2 = pd.read_parquet(fpath+'dsf2.parquet.gzip',
            columns=vars_list)
        df = pd.concat([_df1, _df2], ignore_index=True)
        df = df.query('-2<=exchcd<=3 & shrcd==[10, 11]').copy()
        df = df.drop_duplicates(['permno', 'date'], keep='last')
        df.loc[df['shrout']<=0, 'shrout'] = np.nan
        df.loc[df['vol']<0, 'vol'] = np.nan
        df = df.sort_values(['permno', 'date'], ignore_index=True)
        self.df = df.copy()
       
    def turnover(self, j, min_n, var_name, adjustment=False):
        """
        j: past j-month
        min_n: minimum number of days required to calculate turnover
        var_name: name of variable
        adjustment: NASDAQ volumes adjustment, default False
        """
        df = self.df.copy()
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['yyyymm'] = df['date'].dt.year * 100 + df['date'].dt.month
        df['shrout'] = df['shrout'] * 1000
        df['to_d'] = df['vol'] / df['shrout']
        if adjustment:
            mask1 = (df['date']<'2001-02-01') & (df['exchcd']==3)
            df.loc[mask1, 'to_d'] = (df['vol']/2) / df['shrout']
            mask2 = ((df['date']>='2001-02-01') & (df['date']<='2001-12-31')
                & (df['exchcd']==3))
            df.loc[mask2, 'to_d'] = (df['vol']/1.8) / df['shrout']
            mask3 =  ((df['date']>='2002-01-01') & (df['date']<='2003-12-31')
                & (df['exchcd']==3))
            df.loc[mask3, 'to_d'] = (df['vol']/1.6) / df['shrout']

        _sum = (df.groupby(['permno', 'yyyymm'])
            ['to_d'].sum(min_count=1).to_frame('to_m').reset_index())
        _count = (df.groupby(['permno', 'yyyymm'])
            ['to_d'].count().to_frame('n').reset_index())
        df = _sum.merge(_count, how='inner', on=['permno', 'yyyymm'])
        df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m')
        df['date'] = df['date'] + pd.offsets.MonthEnd(0)
        df['month_idx'] = ((df['date'].dt.year-1925) * 12
            + df['date'].dt.month - 11)
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        df['to_sum'] = (df.groupby('permno')['to_m']
            .rolling(window=j, min_periods=1).sum().reset_index(drop=True))
        df['day_sum'] = (df.groupby('permno')['n']
            .rolling(window=j, min_periods=1).sum().reset_index(drop=True))
        df['l_month_idx'] = df.groupby('permno')['month_idx'].shift(j-1)
        df['_tur'] = df['to_sum'] / df['day_sum']
        df.loc[df['day_sum']<min_n, '_tur'] = np.nan
        df['month_gap'] = df['month_idx'] - df['l_month_idx']
        df.loc[df['month_gap']!=j-1, '_tur'] = np.nan
        df = df.query('_tur==_tur').copy()
        df[var_name] = df['_tur']
        df = df[['permno', 'yyyymm', var_name]]
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df

    def turnover_out(self, j, min_n, var_name, adjustment):
        """
        j: list
            past j-month
        min_n: list
            minimum number of days required to calculate turnover
        var_name: list
            name of variable
        adjustment: list
            NASDAQ volumes adjustment
        """
        df = pd.DataFrame(columns=['permno', 'yyyymm'])
        for k, l, m, n in zip(j, min_n, var_name, adjustment):
            _tmp = self.turnover(k, l, m, n)
            df = df.merge(_tmp, how='outer', on=['permno', 'yyyymm'])

        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df


# Initialize object
db = ap_share_turnover('/Users/ml/Data/wrds/parquet/')

# Turnover using past 6-month daily data without NASDAQ adjustment
tur6 = db.turnover(6, 50, 'tur')

# Turnover using past 6-month daily data with NASDAQ adjustment
tur6_adj = db.turnover(6, 50, 'tur_adj', True)

# Different versions of turnover in one table
tur = db.turnover_out([6, 6], [50, 50], ['tur', 'tur_adj'], [False, True])

