# ------------------------------------------------------------------
#                     Capital gain overhang
#
# Grinblatta and Han (2005)
# Equation 9 (page 319)
# Equation 11 (page 320)
# Fig 2. note
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import time
import os

class ap_cgo:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP daily data
        dsf = conn.raw_sql("""
            select a.permno, a.date, a.prc, a.vol, a.shrout
            from crsp.dsf a left join crsp.msenames b
                on a.permno=b.permno and a.date>=b.namedt and a.date<=b.nameendt
            where b.exchcd between -2 and 3 and b.shrcd between 10 and 11
        """, date_cols=['date'])

        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'Time used (SQL): {(end_time-start_time)/60: 3.1f} mins')

        start_time = time.time()
        dsf = dsf.drop_duplicates(['permno', 'date'], keep='last')
        dsf['shrout'] = dsf['shrout'] * 1000
        dsf['prc'] = dsf['prc'].abs()
        dsf.loc[dsf['prc']<=0, 'prc'] = np.nan
        dsf.loc[dsf['vol']<0, 'vol'] = np.nan
        dsf.loc[dsf['shrout']<=0, 'shrout'] = np.nan
        dsf['permno'] = dsf['permno'].astype(int)
        dsf['weekday'] = dsf['date'].dt.weekday
        self.dsf = dsf.copy()

        end_time = time.time()
        print(f'Time used (clean): {(end_time-start_time)/60: 3.1f} seconds\n')

    def cgo_est(self):
        start_time = time.time()
        df = self.dsf.copy()

        df = df.sort_values(['permno','date'], ignore_index=True)
        df['vol_5day'] = df.groupby('permno')['vol'] \
            .apply(lambda x: pd.Series.rolling(x,window=5).sum())

        df = df.query('weekday==4').copy()
        df['v'] = df['vol_5day'] / df['shrout']
        df.loc[df['v']>=1, 'v'] = np.nan
        df['diff_1v'] = 1 - df['v']
        df['diff_1v'] = np.log(df['diff_1v'])
        df = df.sort_values(['permno','date'], ignore_index=True)
        df['vprod'] = (df.groupby('permno')['diff_1v']
            .rolling(window=259, min_periods=129).sum().reset_index(drop=True))
        df['vprod'] = np.exp(df['vprod'])
        df['v_vprod'] = df['v'] * df['vprod']
        df['k'] = (df.groupby('permno')['v_vprod']
            .rolling(window=259, min_periods=129).sum().reset_index(drop=True))
        df['v_vprod_p'] = df['v_vprod'] * df['prc']
        df = df.sort_values(['permno','date'], ignore_index=True)
        df['r'] = (df.groupby('permno')['v_vprod_p'] \
            .rolling(window=260, min_periods=130).sum().reset_index(drop=True))
        df['r'] = df['r'] / df['k']
        df = df.sort_values(['permno', 'date'], ignore_index=True)
        df['l1prc'] = df.groupby('permno')['prc'].shift(1)

        df = df[['permno', 'date', 'r', 'l1prc']].copy()
        df['cgo'] = (df['l1prc']-df['r']) / df['l1prc']
        df['yyyymm'] = df['date'].dt.year * 100 + df['date'].dt.month
        df = df.groupby(['permno', 'yyyymm'])['cgo'].mean().reset_index()
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- Capital gain overhang ---------')
        print(f'Obs: {len(df)}')
        print(f'Time used: {(end_time-start_time)/60: 3.1f} mins')
        return df

if __name__ == '__main__':
    db = ap_cgo()
    cgo = db.cgo_est()
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    cgo.to_csv(os.path.join(data_dir, 'captial_gain_overhang.txt'),
        sep='\t', index=False)
    print('Done: data is generated')
