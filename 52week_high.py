# ------------------------------------------------------------------
#                       52-week high
#
# week52h: it is the price on last trading day in month t divided by
# the highest price over past 12 months (from month t-11 to t).
#
# Alternative version: skip one month
# week52h_skip: it is the price on last trading day in month t-1 divided
# by the highest price over past 12 months (from month t-12 to t-1).
#
# George and Hwang (2004)
# "The monthly returns of JT’s (6, 6) strategy and the 52-week high
# strategy are obtained the same way. The only difference is that
# stocks are ranked using different measures of past performance
# than industry return. For JT’s strategy, stocks are ranked based
# on their own individual returns over months t-6 to t-1. For the
# 52-week high strategy, stocks are ranked based on P_t−1/high_t−1,
# where P_t−1 is the price of stock i at the end of month t-1 and
# highi,t−1 is the highest price of stock i during the 12-month
# period that ends on the last day of month t-1."
#
# GH (2004) calculate return in month t, so they do not skip one
# month
# JT is Jegadeesh and Titman (1993)
#
# Price is adjusted stock price: prc / cfacpr (CRSP items)
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import os
import time

class ap_week52_high:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP daily data
        dsf = conn.raw_sql("""
            select a.permno, a.date, a.prc, a.cfacpr
            from crsp.dsf a left join crsp.msenames b
                on a.permno=b.permno and a.date>=b.namedt and a.date<=b.nameendt
            where b.exchcd between -2 and 3 and b.shrcd between 10 and 11
        """, date_cols=['date'])

        dsf = dsf.drop_duplicates(['permno', 'date'], keep='last')
        dsf['permno'] = dsf['permno'].astype(int)
        dsf['yyyymm'] = dsf['date'].dt.year*100 + dsf['date'].dt.month
        dsf['prc'] = dsf['prc'].abs()
        dsf.loc[dsf['prc']==0, 'prc'] = np.nan
        dsf.loc[dsf['cfacpr']<=0, 'cfacpr'] = np.nan
        dsf['prc'] = dsf['prc'] / dsf['cfacpr']
        del dsf['cfacpr']
        obs_nonpos = len(dsf.query('prc<=0'))
        self.dsf = dsf.copy()

        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'Obs (non-positive price): {obs_nonpos}')
        print(f'time_used: {(end_time-start_time)/60: 3.1f} mins\n')

    def week52_high(self):
        start_time = time.time()

        # Past 12-month high price for each month
        month_high = (self.dsf.groupby(['permno', 'yyyymm'])
            ['prc'].max().to_frame('month_high').reset_index())
        month_high = month_high.sort_values(['permno', 'yyyymm'],
            ignore_index=True)
        month_high['l1month_high'] = (month_high.groupby('permno')
            ['month_high'].shift(1))
        month_high['pre12high'] = (month_high.groupby('permno')['month_high']
            .rolling(window=12, min_periods=12).max().reset_index(drop=True))
        month_high['pre12high_skip'] = (month_high.groupby('permno')
            ['l1month_high'].rolling(window=12, min_periods=12)
            .max().reset_index(drop=True))
        # Price on last trading day in month t
        month_price = self.dsf.copy()
        month_price = month_price.sort_values(['permno', 'yyyymm', 'date'],
            ignore_index=True)
        month_price = (month_price.drop_duplicates(['permno', 'yyyymm'],
            keep='last').copy())
        del month_price['date']
        month_price = month_price.sort_values(['permno', 'yyyymm'],
            ignore_index=True)
        month_price['l1prc'] = month_price.groupby('permno')['prc'].shift(1)

        df = month_price.merge(month_high, how='inner', on=['permno', 'yyyymm'])
        df['week52h'] = df['prc'] / df['pre12high'] - 1
        df['week52h_skip'] = df['l1prc'] / df['pre12high_skip'] - 1
        df = df[['permno', 'yyyymm', 'week52h', 'week52h_skip']]
        obs_with_missing = len(df)
        df = df.dropna(subset=['week52h', 'week52h_skip'], how='all')
        obs = len(df)
        start_month = df['yyyymm'].min()
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print('--------- 52-week high estimation ---------')
        print(f'Obs with all missing: {obs_with_missing}')
        print(f'Obs: {obs}')
        print(f'Start month: {start_month}')
        print(f'time_used: {(end_time-start_time)/60: 3.1f} mins\n')
        return df

if __name__ == '__main__':
    db = ap_week52_high()
    week52 = db.week52_high()
    week52 = week52.sort_values(['permno', 'yyyymm'], ignore_index=True)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    week52.to_csv(os.path.join(data_dir, 'week52_high.txt'),
        sep='\t', index=False)
    print('Done: data is generated')
