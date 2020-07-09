# ----------------------------------------------------------------
#                   Idiosyncratic volatility
#
# Idiosyncratic volatility is defined as the standard deviation
# of residuals from asset pricing models. There are three models
# available: CAPM, Fama-French 3-factor and Fama-French 5-factor.
# Regression is estimated in each month and at least 15 days are
# required.
# If use the entire CRSP daily data, there will be around 3.5
# millions regressions. To speed up the estimation,
# multiprocessing is available and you can set the number of CPUs 
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import multiprocessing as mp
import datetime


class ap_ivol:
    def __init__(self, fpath):
        vars_list = ['permno', 'date', 'ret', 'exchcd', 'shrcd']
        _df1 = pd.read_parquet(fpath+'dsf1.parquet.gzip', columns=vars_list)
        _df2 = pd.read_parquet(fpath+'dsf2.parquet.gzip', columns=vars_list)
        df = pd.concat([_df1, _df2], ignore_index=True)
        df = df.query('-2<=exchcd<=3 & shrcd==[10, 11]').copy()
        df = df.drop_duplicates(['permno', 'date'], keep='last')
        df.loc[df['ret']<=-1, 'ret'] = np.nan
        df = df.drop(['exchcd', 'shrcd'], axis=1)
        df = df.sort_values(['permno', 'date'], ignore_index=True)
        ff3 = pd.read_csv('/Users/ml/Data/ff/ff3_daily.csv')
        for i in ff3.columns[1:]:
            ff3[i] = ff3[i] / 100

        ff5 = pd.read_csv('/Users/ml/Data/ff/ff5_daily.csv')
        for i in ff5.columns[1:]:
            ff5[i] = ff5[i] / 100

        self.df = df.copy()
        self.ff3 = ff3
        self.ff5 = ff5

    def ols_b(self, data, x_var, y_var):
        x = data[x_var]
        x.loc[:, 'a'] = 1
        x = x.to_numpy()
        y = data[y_var].to_numpy()
        b = np.linalg.inv(x.T@x) @ x.T @ y
        return b

    def df_apply(self, args):
        data, factors = args
        grouped = (data.groupby(['permno', 'yyyymm'])
            .apply(self.ols_b, x_var=factors, y_var='retx').reset_index())
        return grouped

    def ivolatility(self, model, outvar, ncpu):
        '''
        model: string
            'market', 'ff3', 'ff5'
        outvar: string
            name for calculated variable
        ncpu: int
            number of cpu
        '''
        start_time = datetime.datetime.now()
        df = self.df.copy()
        if model == 'market':
            factors = ['mktrf']
            df = df.merge(self.ff3, how='left', on='date')
        elif model == 'ff3':
            factors = ['mktrf', 'smb', 'hml']
            df = df.merge(self.ff3, how='left', on='date')
        elif model == 'ff5':
            factors = ['mktrf', 'smb', 'hml', 'rmw', 'cma']
            df = df.merge(self.ff5, how='left', on='date')

        df['retx'] = df['ret'] - df['rf']
        df['yyyymm'] = (df['date']/100).astype(int)
        df = df.dropna()
        df['n'] = df.groupby(['permno', 'yyyymm'])['retx'].transform('count')
        df['std'] = df.groupby(['permno', 'yyyymm'])['retx'].transform('std')
        df = df.query('n>=15 & std>0').copy()
        pool = mp.Pool(processes=ncpu)
        permno_list = df['permno'].unique()
        permno_split = np.array_split(permno_list, ncpu)
        _b = pool.map(self.df_apply,
            [(df.query('permno==list(@i)'), factors) for i in permno_split])
        pool.close()
        pool.join()
        b = pd.concat(_b)
        df = df.merge(b, how='inner', on=['permno', 'yyyymm'])
        if model == 'market':
            for i, j in zip(['a','b1'], [1,0]):
                df[i] = df[0].apply(lambda x: x[j])

            df['p'] = df['a'] + df['b1']*df['mktrf']
            df['resid'] = df['retx'] - df['p']
        elif model == 'ff3':
            for i, j in zip(['a','b1','b2','b3'], [3,0,1,2]):
                df[i] = df[0].apply(lambda x: x[j])

            df['p'] = (df['a'] + df['b1']*df['mktrf']
                + df['b2']*df['smb'] + df['b3']*df['hml'])
            df['resid'] = df['retx'] - df['p']
        elif model == 'ff5':
            for i, j in zip(['a','b1','b2','b3','b4','b5'], [5,0,1,2,3,4]):
                df[i] = df[0].apply(lambda x: x[j])

            df['p'] = (df['a'] + df['b1']*df['mktrf'] + df['b2']*df['smb']
                + df['b3']*df['hml'] + df['b4']*df['rmw'] + df['b5']*df['cma']) 
            df['resid'] = df['retx'] - df['p']

        df = (df.groupby(['permno', 'yyyymm'])['resid']
            .std().to_frame(outvar).reset_index())
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        end_time = datetime.datetime.now()
        print('start_time: %s' %start_time)
        print('end_time: %s' %end_time)
        print('number of stocks: %s' %len(permno_list))
        print('number of regressions: %s' %len(df))
        return df

db = ap_ivol('/Users/ml/Data/wrds/parquet/')

ivol_ff3 = db.ivolatility('ff3', 'ivol_ff3', 5)

