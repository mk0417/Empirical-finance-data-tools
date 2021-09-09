# ------------------------------------------------------------------
#                           Skewness
#
# Idiosyncratic skewness is the skewness of residuals from regression
# of stock excess return on market excess return in each month
# (require at least 15 days)
#
# Coskewness is estimated as the formula below
# cs = E[e_i,(e_m)^2] / (sqrt(E[(e_i)^2])*E[(e_m)^2])
# e_i is the residual from regression of stock excess return on market
# excess return. e_m is the demeaned market excess return. This is
# estimated in each month (require at least 15 days)
#
# Hou, Xue and Zhang (2020)
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import time
import os

class ap_skew:
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

        # Extract factor data
        # Data is available from 1926-07-01
        mktrf = conn.raw_sql("""
            select date, mktrf, rf
            from ff.factors_daily
            order by date
        """, date_cols=['date'])
        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'Time used (SQL): {(end_time-start_time)/60: 3.1f} mins')

        start_time = time.time()
        dsf = dsf.drop_duplicates(['permno', 'date'], keep='last')
        dsf.loc[dsf['ret']<=-1, 'ret'] = np.nan
        dsf['permno'] = dsf['permno'].astype(int)
        self.dsf = dsf.copy()
        self.mktrf = mktrf.copy()

        end_time = time.time()
        print(f'Time used (clean): {(end_time-start_time)/60: 3.1f} seconds\n')

    # Compute covariance of X and Y
    # This will be used to estimate regression coefficients
    # See the link below if you need to check the formula
    # https://en.wikipedia.org/wiki/Simple_linear_regression
    def cov_m(self, data):
        x = data['mktrf'].to_numpy()
        y = data['retx'].to_numpy()
        res = ((x-np.mean(x)) @ (y-np.mean(y))) / (len(x) - 1)
        return res

    def skew_est(self):
        start_time = time.time()
        df = self.dsf.merge(self.mktrf, how='inner', on='date')

        df['retx'] = df['ret'] - df['rf']
        del df['ret']
        df['yyyymm'] = df['date'].dt.year * 100 + df['date'].dt.month
        df = df.dropna()
        df['n_day'] = (df.groupby(['permno', 'yyyymm'])
            ['retx'].transform('count'))
        df['std'] = df.groupby(['permno', 'yyyymm'])['retx'].transform('std')
        # Require at least 15 days
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

        # Residual
        df = df.merge(b, how='inner', on=['permno', 'yyyymm'])
        df['e'] = df['retx'] - (df['a'] + df['b']*df['mktrf'])
        # Idiosyncratic skewness
        iskew = (df.groupby(['permno', 'yyyymm'])['e']
            .skew().to_frame('iskew').reset_index())
        # Coskewness
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
        df = e_mktrf_dm2.join(e2, how='inner').join(mktrf_dm2, how='inner')
        df = df.reset_index()
        df['coskew'] = df['e_mktrf_dm2'] / (np.sqrt(df['e2'])*df['mktrf_dm2'])
        df = df[['permno', 'yyyymm', 'coskew']].copy()
        df = df.merge(iskew, how='outer', on=['permno', 'yyyymm'])
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- Skewness ---------')
        print(f'Obs: {len(df)}')
        print(f'Time used: {(end_time-start_time)/60: 3.1f} mins')
        return df

if __name__ == '__main__':
    db = ap_skew()
    sk = db.skew_est()
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    sk.to_csv(os.path.join(data_dir, 'skewness.txt'), sep='\t', index=False)
    print('Done: data is generated')
