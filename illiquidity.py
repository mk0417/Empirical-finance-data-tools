# ----------------------------------------------------------------------
#                          Illiquidity
# 
# Amihud illquidity is defined as average daily return-to-volume over
# past n month. Return-to-volume is ratio of absolute daily return to
# dollar trading volume (stock price times number of shares traded).
# By default, there is no adjustment of NASDAQ volumes. Please set
# `adjustment` to True to adjust NASDAQ volumes.
# ----------------------------------------------------------------------

import pandas as pd
import numpy as np


class ap_illiquidity:
    def __init__(self, fpath):
        vars_list = ['permno', 'date', 'exchcd', 'shrcd', 'ret', 'prc', 'vol']
        _df1 = pd.read_parquet(fpath+'dsf1.parquet.gzip', columns=vars_list)
        _df2 = pd.read_parquet(fpath+'dsf2.parquet.gzip', columns=vars_list)
        df = pd.concat([_df1, _df2], ignore_index=True)
        df = df.query('-2<=exchcd<=3 & shrcd==[10, 11]').copy()
        df = df.drop_duplicates(['permno', 'date'], keep='last')
        df['price'] = df['prc'].abs()
        df.loc[df['price']<=0, 'price'] = np.nan
        df.loc[df['vol']<=0, 'vol'] = np.nan
        df.loc[df['ret']<=-1, 'ret'] = np.nan
        df = df.sort_values(['permno', 'date'], ignore_index=True)
        self.df = df.copy()
 
    def illiquidity(self, j, min_n, var_name, adjustment=False):
        df = self.df[['permno', 'date', 'price', 'ret', 'vol', 'exchcd']].copy()
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['yyyymm'] = df['date'].dt.year * 100 + df['date'].dt.month
        df['ami_d'] = df['ret'].abs() / (df['price']*df['vol'])
        if adjustment:
            mask1 = (df['date']<'2001-02-01') & (df['exchcd']==3)
            df.loc[mask1, 'ami_d'] = (df['ret'].abs()
                / (df['price']*(df['vol']/2)))
            mask2 = ((df['date']>='2001-02-01') & (df['date']<='2001-12-31')
                & (df['exchcd']==3))
            df.loc[mask2, 'ami_d'] = (df['ret'].abs()
                / (df['price']*(df['vol']/1.8)))
            mask3 =  ((df['date']>='2002-01-01') & (df['date']<='2003-12-31')
                & (df['exchcd']==3)) 
            df.loc[mask3, 'ami_d'] = (df['ret'].abs()
                / (df['price']*(df['vol']/1.6)))

        _sum = (df.groupby(['permno', 'yyyymm'])
            ['ami_d'].sum(min_count=1).to_frame('ami_m').reset_index())
        _count = (df.groupby(['permno', 'yyyymm'])
            ['ami_d'].count().to_frame('n').reset_index())
        df = _sum.merge(_count, how='inner', on=['permno', 'yyyymm'])
        df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m')
        df['date'] = df['date'] + pd.offsets.MonthEnd(0)
        df['month_idx'] = ((df['date'].dt.year-1925) * 12
            + df['date'].dt.month - 11)
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        df['ami_sum'] = (df.groupby('permno')['ami_m']
            .rolling(window=j, min_periods=1).sum().reset_index(drop=True))
        df['day_sum'] = (df.groupby('permno')['n']
            .rolling(window=j, min_periods=1).sum().reset_index(drop=True))
        df['l_month_idx'] = df.groupby('permno')['month_idx'].shift(j-1)
        df['_ami'] = df['ami_sum'] / df['day_sum']
        df.loc[df['day_sum']<min_n, '_ami'] = np.nan
        df['month_gap'] = df['month_idx'] - df['l_month_idx']
        df.loc[df['month_gap']!=j-1, '_ami'] = np.nan
        df = df.query('_ami==_ami').copy()
        df[var_name] = df['_ami']
        df = df[['permno', 'yyyymm', var_name]]
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df

    def illiquidity_out(self, j, min_n, var_name, adjustment):
        '''
        j: list
            past j-month
        min_n: list
            minimum number of days required to calculate turnover
        var_name: list
            name of variable
        adjustment: list
            NASDAQ volumes adjustment
        '''
        df = pd.DataFrame(columns=['permno', 'yyyymm'])
        for k, l, m, n in zip(j, min_n, var_name, adjustment):
            _tmp = self.illiquidity(k, l, m, n)
            df = df.merge(_tmp, how='outer', on=['permno', 'yyyymm'])

        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df

db = ap_illiquidity('/Users/ml/Data/wrds/parquet/')

ami = db.illiquidity_out([6, 6], [50, 50], ['ami', 'ami_adj'], [False, True])

