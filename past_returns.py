# -------------------------------------------------------------------------
#                       Past n-month return
#
# Calculate return for past n-month
# For example, you can calculate past 6-month return by setting
# `j` (horizon )to 6
# For momentum, literature usually skip the most recent month when
# computing past returns, and this feature is set by default. Please
# set `skip` to False if you do not want to skip
# -------------------------------------------------------------------------


import pandas as pd
import numpy as np


class ap_pre_ret:
    def __init__(self, fpath):
        self.fpath = fpath
        vars_list = ['permno', 'date', 'ret',
            'prc', 'shrout', 'exchcd', 'shrcd']
        df = pd.read_parquet(self.fpath+'msf.parquet.gzip', columns=vars_list)
        df = df.query('-2<=exchcd<=3 & shrcd==[10, 11]').copy()
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['date'] = df['date'] + pd.offsets.MonthEnd(0)
        df['yyyymm'] = df['date'].dt.year * 100 + df['date'].dt.month
        df = df.drop_duplicates(['permno', 'yyyymm'], keep='last')
        df['price'] = df['prc'].abs()
        df.loc[df['price']<=0, 'price'] = np.nan
        df['me'] = df['price'] * df['shrout'] / 1000
        df.loc[df['me']<=0, 'me'] = np.nan
        df['month_idx'] = ((df['date'].dt.year-1925) * 12
            + df['date'].dt.month - 11)
        df.loc[df['ret']<=-1, 'ret'] = np.nan
        df = df.drop('prc', axis=1).copy()
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        self.df = df.copy()

    def pre_ret(self, j, skip=True):
        '''
        j: formation periods: past j month
        skip: bool, default True
            True to skip most recent month
            False to include most recent month
        '''
        df = self.df.copy()
        df['logret'] = np.log(1+df['ret'])
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        df['l_logret'] = df.groupby('permno')['logret'].shift(1)
        if skip:
            df['pre'+str(j)+'ret'] = (df.groupby('permno')['l_logret']
                .rolling(window=j, min_periods=j-1)
                .sum().reset_index(drop=True))
            df['l_month_idx'] = df.groupby('permno')['month_idx'].shift(j)
            df['month_gap'] = df['month_idx'] - df['l_month_idx']
            df['pre'+str(j)+'ret'] = np.exp(df['pre'+str(j)+'ret']) - 1
            df.loc[df['month_gap']!=j, 'pre'+str(j)+'ret'] = np.nan
            df = df[df['pre'+str(j)+'ret'].notna()].copy()
            df = df[['permno', 'yyyymm', 'pre'+str(j)+'ret']]
        else:
            df['pre'+str(j)+'ret_noskip'] = (df.groupby('permno')['logret']
                .rolling(window=j, min_periods=j-1)
                .sum().reset_index(drop=True))
            df['l_month_idx'] = df.groupby('permno')['month_idx'].shift(j-1)
            df['month_gap'] = df['month_idx'] - df['l_month_idx']
            df['pre'+str(j)+'ret_noskip'] = (
                np.exp(df['pre'+str(j)+'ret_noskip']) - 1)
            df.loc[df['month_gap']!=j-1, 'pre'+str(j)+'ret_noskip'] = np.nan
            df = df[df['pre'+str(j)+'ret_noskip'].notna()].copy()
            df = df[['permno', 'yyyymm', 'pre'+str(j)+'ret_noskip']]
        return df

    def pre_ret_out(self, j, skip):
        '''
        j: list
            formation periods: past j month
        skip: list
            True to skip most recent month
            False to include most recent month
        '''
        df = pd.DataFrame(columns=['permno', 'yyyymm'])
        for m, n in zip(j, skip):
            _tmp = self.pre_ret(m, n)
            df = df.merge(_tmp, how='outer', on=['permno', 'yyyymm'])

        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df


# Initialize object
db = ap_pre_ret('/Users/ml/Data/wrds/parquet/')

# Past 6-month return skipping most recent month
pre6ret = db.pre_ret(6)

# Past 6-month return including most recent month
pre6ret_noskip = db.pre_ret(6, False)

# Different formation horizons
pret = db.pre_ret_out([6, 12, 6, 12], [True, True, False, False])

