# --------------------------------
# Conditional skewness
# --------------------------------

import pandas as pd
import numpy as np
import datetime


class ap_coskew:
    def __init__(self, fpath):
        vars_list = ['permno', 'date', 'ret', 'exchcd', 'shrcd']
        _df1 = pd.read_parquet(fpath+'dsf1.parquet.gzip', columns=vars_list)
        _df2 = pd.read_parquet(fpath+'dsf2.parquet.gzip', columns=vars_list)
        df = pd.concat([_df1, _df2], ignore_index=True)
        df = df.query('-2<=exchcd<=3 & shrcd==[10, 11]').copy()
        df = df.drop_duplicates(['permno', 'date'], keep='last')
        df.loc[df['ret']<=-1, 'ret'] = np.nan
        df = df.sort_values(['permno', 'date'], ignore_index=True)
        ff = pd.read_csv('/Users/ml/Data/ff/ff3_daily.csv')
        for i in ff.columns[1:]:
            ff[i] = ff[i] / 100

        self.df = df.copy()
        self.ff = ff.copy()

    def cov_m(self, data):
        x = data['mktrf'].to_numpy()
        y = data['retx'].to_numpy()
        res = ((x-np.mean(x)) @ (y-np.mean(y))) / (len(x) - 1)
        return res

    def coskew(self):
        start_time = datetime.datetime.now()
        df = self.df[['permno', 'date', 'ret']].copy()
        mktrf = self.ff[['date', 'mktrf', 'rf']].copy()
        df = df.merge(mktrf, how='inner', on='date')
        df['retx'] = df['ret'] - df['rf']
        df['yyyymm'] = (df['date']/100).astype(int)
        df = df.dropna()
        df['n_day'] = (df.groupby(['permno', 'yyyymm'])
            ['retx'].transform('count'))
        df['std'] = df.groupby(['permno', 'yyyymm'])['retx'].transform('std')
        df = df.query('n_day>=15 & std>0').copy()
        covar = (df.groupby(['permno', 'yyyymm'])
            .apply(self.cov_m).to_frame('covar'))
        var = df.groupby(['permno', 'yyyymm'])['mktrf'].var().to_frame('var')
        y_bar = (df.groupby(['permno', 'yyyymm'])
            ['retx'].mean().to_frame('y_bar'))
        x_bar = (df.groupby(['permno', 'yyyymm'])
            ['mktrf'].mean().to_frame('x_bar'))
        b = (covar.join(var, how='inner').join(y_bar, how='inner')
            .join(x_bar, how='inner').reset_index())
        b['b'] = b['covar'] / b['var']
        b['a'] = b['y_bar'] - (b['b']*b['x_bar'])
        b = b[['permno','yyyymm', 'a', 'b']].copy()
        df = df.merge(b, how='inner', on=['permno', 'yyyymm'])
        df['e'] = df['retx'] - (df['a'] + df['b']*df['mktrf'])
        df['mktrf_dm'] = (df.groupby(['permno', 'yyyymm'])['mktrf']
            .transform(lambda x: x-x.mean()))
        df['e_mktrf_dm2'] = df['e'] * (df['mktrf_dm']**2)
        df['e2'] = df['e']**2
        df['mktrf_dm2'] = df['mktrf_dm']**2
        e_mktrf_dm2 = (df.groupby(['permno', 'yyyymm'])
            ['e_mktrf_dm2'].mean().to_frame())
        e2 = df.groupby(['permno', 'yyyymm'])['e2'].mean().to_frame()
        mktrf_dm2 = (df.groupby(['permno', 'yyyymm'])
            ['mktrf_dm2'].mean().to_frame())
        cs = e_mktrf_dm2.join(e2, how='inner').join(mktrf_dm2, how='inner')
        cs = cs.reset_index()
        cs['cs'] = cs['e_mktrf_dm2'] / (np.sqrt(cs['e2'])*cs['mktrf_dm2'])
        cs = cs[['permno', 'yyyymm', 'cs']].copy()
        cs = cs.sort_values(['permno', 'yyyymm'], ignore_index=True)
        end_time = datetime.datetime.now()
        print('start time: %s' %start_time)
        print('end time: %s' %end_time)
        return cs
    
db = ap_coskew('/Users/ml/Data/wrds/parquet/')

cs = db.coskew()
