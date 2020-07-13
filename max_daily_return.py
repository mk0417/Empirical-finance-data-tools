# ------------------------------------------------------------------
#                       Maximum daily return
#
# Maximum daily return is defined as the average of largest n
# daily return in a month (at least 15 daily returns are required)
# ------------------------------------------------------------------

import pandas as pd
import numpy as np

class ap_max_ret:
    def __init__(self, fpath):
        vars_list = ['permno', 'date', 'ret','exchcd', 'shrcd']
        _df1 = pd.read_parquet(fpath+'dsf1.parquet.gzip', columns=vars_list)
        _df2 = pd.read_parquet(fpath+'dsf2.parquet.gzip', columns=vars_list)
        df = pd.concat([_df1, _df2], ignore_index=True)
        df = df.query('-2<=exchcd<=3 & shrcd==[10, 11]').copy()
        df = df.drop_duplicates(['permno', 'date'], keep='last')
        df.loc[df['ret']<=-1, 'ret'] = np.nan
        self.df = df.copy()

    def max_ret(self, n):
        df = self.df[['permno', 'date', 'ret']].copy()
        df['yyyymm'] = (df['date']/100).astype(int)
        df['n_day'] = df.groupby(['permno', 'yyyymm'])['ret'].transform('count')
        df = df.query('n_day>=15').copy()
        df['rank'] = (df.groupby(['permno', 'yyyymm'])['ret']
            .rank(method='min', ascending=False))
        df = df.query('rank<=@n').copy()
        df = (df.groupby(['permno', 'yyyymm'])['ret']
            .mean().to_frame('mdr'+str(n)).reset_index())
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df

    def max_ret_out(self, n):
        """
        n: list
            highest n days
        """
        df = pd.DataFrame(columns=['permno', 'yyyymm'])
        for i in n:
            _tmp = self.max_ret(i)
            df = df.merge(_tmp, how='outer', on=['permno', 'yyyymm'])

        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df

db = ap_max_ret('/Users/ml/Data/wrds/parquet/')

mdr = db.max_ret_out([1, 5])

