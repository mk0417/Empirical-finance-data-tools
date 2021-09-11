# ------------------------------------------------------------------
#                       Total volatility
#
# Total volatility is the standard deviation of daily returns in a
# month with at least 15 daily returns
#
# Ang, Hodrick, Xing and Zhang (2006)
# Hou, Xue and Zhang (2020)
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import os
import time

class ap_tvol:
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
        self.dsf = dsf.copy()

        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'Time used: {(end_time-start_time)/60: 3.1f} mins\n')

    def tvol_est(self):
        start_time = time.time()
        df = self.dsf.copy()

        df['yyyymm'] = df['date'].dt.year*100 + df['date'].dt.month
        # Require at least 15 days in a month
        df = df.dropna()
        df['n'] = (df.groupby(['permno', 'yyyymm'])['ret'].transform('count'))
        df = df.query('n>=15').copy()
        del df['n']
        df = (df.groupby(['permno', 'yyyymm'])
            ['ret'].std().to_frame('tvol').reset_index())
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- Total volatility ---------\n')
        print(f"Percent (zero std): {len(df.query('tvol==0'))/len(df): 3.2%}")
        print(f'Time used: {end_time-start_time: 3.1f} seconds\n')
        return df

if __name__ == '__main__':
    db = ap_tvol()
    tvol = db.tvol_est()
    tvol = tvol.sort_values(['permno', 'yyyymm'], ignore_index=True)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    tvol.to_csv(os.path.join(data_dir, 'tvol.txt'), sep='\t', index=False)
    print('Done: data is generated')
