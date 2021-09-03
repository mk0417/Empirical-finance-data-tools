# ----------------------------------------------------------------
#                   Idiosyncratic volatility
#
# Idiosyncratic volatility is defined as the standard deviation
# of residuals from asset pricing models. There are two models
# available: CAPM and Fama-French 3-factor. Regression is
# estimated in each month and at least 15 days are required.
# If use the entire CRSP daily data, there will be more than 3.5
# millions regressions from. To speed up the estimation,
# multiprocessing is applied.
# ----------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
import itertools
import time
import os

class ap_ivol:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP daily data
        dsf = conn.raw_sql("""
            select a.permno, a.date, a.ret
            from crsp.dsf a left join crsp.msenames b
                on a.permno=b.permno and a.date>=b.namedt and a.date<=b.nameendt
            where b.exchcd between -2 and 3 and b.shrcd between 10 and 11
        """, date_cols=['date'])

        dsf = dsf.drop_duplicates(['permno', 'date'], keep='last')
        dsf.loc[dsf['ret']<=-1, 'ret'] = np.nan
        dsf['permno'] = dsf['permno'].astype(int)

        # Extract factor data
        # Data is available from 1926-07-01
        ff3 = conn.raw_sql("""
            select date, mktrf, smb, hml, rf
            from ff.factors_daily
            order by date
        """, date_cols=['date'])

        self.dsf = dsf.copy()
        self.ff3 = ff3.copy()

        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'time_used: {(end_time-start_time)/60: 3.1f} mins\n')

    def ols_b(self, data, x_var, y_var):
        x = data[x_var].copy()
        x.loc[:, 'a'] = 1
        x = x.to_numpy()
        y = data[y_var].to_numpy()
        b = np.linalg.inv(x.T@x) @ x.T @ y
        return b

    def groupby_ols(self, data, x_var, i, l_res):
        df = data.query('permno==list(@i)').copy()
        df = (df.groupby(['permno', 'yyyymm'])
            .apply(self.ols_b, x_var=x_var, y_var='retx').reset_index())
        df = tuple(df.itertuples(index=False, name=None))
        l_res.append(df)

    def ivol_est(self, model, outvar):
        start_time = time.time()
        df = self.dsf.copy()

        if model == 'capm':
            factors = ['mktrf']
            df = df.merge(self.ff3, how='left', on='date')
        elif model == 'ff3':
            factors = ['mktrf', 'smb', 'hml']
            df = df.merge(self.ff3, how='left', on='date')

        df['retx'] = df['ret'] - df['rf']
        df['yyyymm'] = df['date'].dt.year*100 + df['date'].dt.month
        # Require at least 15 days in a month
        # Require standard deviation is greater than 0 to make sure the daily
        # returns in a month are not the same
        df = df.dropna()
        df['n'] = df.groupby(['permno', 'yyyymm'])['retx'].transform('count')
        df['std'] = df.groupby(['permno', 'yyyymm'])['retx'].transform('std')
        df = df.query('n>=15 & std>0').copy()
        df = df.drop(columns=['ret', 'rf', 'n', 'std'])
        df = df.sort_values(['permno', 'date'], ignore_index=True)

        # Python is slow when running large number of regressions by group
        # Parallel to speed up
        # TODO: pure numpy should be faster than pandas
        permno_list = df['permno'].unique()
        permno_split = np.array_split(permno_list, 7)
        manager = mp.Manager()
        l_res = manager.list()
        with parallel_backend('loky', n_jobs=mp.cpu_count()-1):
            Parallel()(delayed(self.groupby_ols)(df, factors, i, l_res)
                for i in permno_split)

        # Flatten list and generate dataframe to improve performance
        # This is faster than appending dataframes
        l_res = list(itertools.chain.from_iterable(list(l_res)))
        b = pd.DataFrame(l_res)
        b.columns = ['permno', 'yyyymm', 'est']
        res_df = df.merge(b, how='inner', on=['permno', 'yyyymm'])
        res_df = res_df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        if model == 'capm':
            for i, j in zip(['a', 'b1'], [1, 0]):
                res_df[i] = res_df['est'].apply(lambda x: x[j])

            res_df['p'] = res_df['a'] + res_df['b1']*res_df['mktrf']
        elif model == 'ff3':
            for i, j in zip(['a', 'b1', 'b2', 'b3'], [3, 0, 1, 2]):
                res_df[i] = res_df['est'].apply(lambda x: x[j])

            res_df['p'] = (res_df['a'] + res_df['b1']*res_df['mktrf']
                + res_df['b2']*res_df['smb'] + res_df['b3']*res_df['hml'])

        res_df['resid'] = res_df['retx'] - res_df['p']
        res_df = (res_df.groupby(['permno', 'yyyymm'])['resid']
            .std().to_frame(outvar).reset_index())
        res_df = res_df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- IVOL estimation: {model} ---------')
        print(f'time_used: {(end_time-start_time)/60: 3.1f} mins')
        print(f'number of stocks: {len(permno_list)}')
        print(f'number of regressions: {len(res_df)}\n')
        return res_df

if __name__ == '__main__':
    db = ap_ivol()
    ivol_capm = db.ivol_est('capm', 'ivol_capm')
    ivol_ff3 = db.ivol_est('ff3', 'ivol_ff3')
    ivol = ivol_capm.merge(ivol_ff3, how='left', on=['permno', 'yyyymm'])
    ivol = ivol.sort_values(['permno', 'yyyymm'], ignore_index=True)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    ivol.to_csv(os.path.join(data_dir, 'ivol.txt'), sep='\t', index=False)
    print('Done: data is generated')
