# ------------------------------------------------------------------
#                       Maximum daily return
#
# Maximum daily return is defined as the average of largest n daily
# return in a month (at least 15 daily returns are required)
#
# Bali, Cakici and Whitelaw (2011)
# "While conditioning on the single day with the maximum return is
# both simple and intuitive as a proxy for extreme positive returns,
# it is also slightly arbitrary. As an alternative, we also rank
# stocks by the average of the N (N=1, 2, …, 5) highest daily
# returns within the month"
#
# For example, mdr1 is the maximum return in a month. And mdr2 is
# the average of top 2 daily returns in a month.
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import os
import time

class ap_maxret:
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

        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'time_used: {(end_time-start_time)/60: 3.1f} mins\n')

        # Rank returns to find top 5 returns in a month
        start_time = time.time()
        dsf['yyyymm'] = dsf['date'].dt.year*100 + dsf['date'].dt.month
        # Require at least 15 days in a month
        dsf = dsf.dropna()
        dsf['n_day'] = (dsf.groupby(['permno', 'yyyymm'])
            ['ret'].transform('count'))
        dsf = dsf.query('n_day>=15').copy()
        del dsf['n_day']
        dsf['rank'] = (dsf.groupby(['permno', 'yyyymm'])['ret']
            .rank(method='min', ascending=False))
        self.dsf = dsf.copy()

        end_time = time.time()
        print(f'--------- Rank returns ---------')
        print(f'time_used: {end_time-start_time: 3.1f} seconds\n')

    def maxret(self, n):
        start_time = time.time()
        df = self.dsf.query('rank<=@n').copy()
        df = (df.groupby(['permno', 'yyyymm'])['ret']
            .mean().to_frame('mdr'+str(n)).reset_index())
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- MDR{n} ---------')
        print(f'time_used: {end_time-start_time: 3.1f} seconds\n')
        return df

if __name__ == '__main__':
    db = ap_maxret()
    mdr = pd.DataFrame(columns=['permno', 'yyyymm'])
    for i in range(1, 6):
        _tmp = db.maxret(i)
        mdr = mdr.merge(_tmp, how='left', on=['permno', 'yyyymm'])

    mdr = mdr.sort_values(['permno', 'yyyymm'], ignore_index=True)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    mdr.to_csv(os.path.join(data_dir, 'mdr.txt'), sep='\t', index=False)
    print('Done: data is generated')
