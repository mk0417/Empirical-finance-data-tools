# -------------------------------------------------------------------------
#                       Past n-month return
#
# pre3ret: past 3-month return
# pre6ret: past 6-month return
# pre9ret: past 9-month return
# pre12ret: past 12-month return
# pre12_7ret: past 12 to 7 month return
#
# All calculations skip current month
# Require at least n-1 months in past n months
# -------------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import os
import time

class ap_preret:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP daily data
        msf = conn.raw_sql("""
            select a.permno, a.date, a.ret
            from crsp.msf a left join crsp.msenames b
                on a.permno=b.permno and a.date>=b.namedt and a.date<=b.nameendt
            where b.exchcd between -2 and 3 and b.shrcd between 10 and 11
        """, date_cols=['date'])

        msf['permno'] = msf['permno'].astype(int)
        msf['date'] = msf['date'] + pd.offsets.MonthEnd(0)
        msf['yyyymm'] = msf['date'].dt.year*100 + msf['date'].dt.month
        msf = msf.drop_duplicates(['permno', 'yyyymm'])
        msf['midx'] = ((msf['date'].dt.year-1925) * 12
            + msf['date'].dt.month - 11)
        msf.loc[msf['ret']<=-1, 'ret'] = np.nan
        msf = msf.sort_values(['permno', 'yyyymm'], ignore_index=True)
        self.msf = msf.copy()

        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'time_used: {end_time-start_time: 3.1f} seconds\n')

    def preret_est(self, j):
        start_time = time.time()
        df = self.msf.copy()

        df['logret'] = np.log(1+df['ret'])
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        df['l1logret'] = df.groupby('permno')['logret'].shift(1)
        # Past n-month returns with at least n-1 months
        df['pre'+str(j)+'ret'] = (df.groupby('permno')['l1logret']
            .rolling(window=j, min_periods=j-1)
            .sum().reset_index(drop=True))
        df['lmidx'] = df.groupby('permno')['midx'].shift(j)
        df['month_gap'] = df['midx'] - df['lmidx']
        df['pre'+str(j)+'ret'] = np.exp(df['pre'+str(j)+'ret']) - 1
        # 137 obs with month gaps
        # Set to missing if there is month gap
        df.loc[df['month_gap']!=j, 'pre'+str(j)+'ret'] = np.nan
        df = df[df['pre'+str(j)+'ret'].notna()].copy()
        df = df[['permno', 'yyyymm', 'pre'+str(j)+'ret']]
        obs = len(df)
        start_month = df['yyyymm'].min()
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- Past {j}-month return ---------')
        print(f'Obs: {obs}')
        print(f'Start month: {start_month}')
        print(f'time_used: {end_time-start_time: 3.1f} seconds')
        return df

    def pre12_7ret_est(self):
        start_time = time.time()
        print('\n--------- Past 12-to-7 return ---------')
        pre6ret = self.preret_est(6)
        pre12ret = self.preret_est(12)

        df = pre6ret.merge(pre12ret, how='inner', on=['permno', 'yyyymm'])
        df['pre12_7ret'] = np.log(1+df['pre12ret']) - np.log(1+df['pre6ret'])
        df['pre12_7ret'] = np.exp(df['pre12_7ret']) - 1
        df = (df.query('pre12_7ret==pre12_7ret')
            [['permno', 'yyyymm', 'pre12_7ret']].copy())
        obs = len(df)
        start_month = df['yyyymm'].min()
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'Obs: {obs}')
        print(f'Start month: {start_month}')
        print(f'time_used: {end_time-start_time: 3.1f} seconds\n')
        return df

if __name__ == '__main__':
    db = ap_preret()
    preret = db.preret_est(3)
    for i in [6, 9, 12]:
        tmp = db.preret_est(i)
        preret = preret.merge(tmp, how='left', on=['permno', 'yyyymm'])

    tmp = db.pre12_7ret_est()
    preret = preret.merge(tmp, how='left', on=['permno', 'yyyymm'])
    preret = preret.sort_values(['permno', 'yyyymm'], ignore_index=True)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    preret.to_csv(os.path.join(data_dir, 'preret.txt'), sep='\t', index=False)
    print('Done: data is generated')

