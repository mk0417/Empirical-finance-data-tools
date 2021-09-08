# ----------------------------------------------------------------------
#                            Volume
#
# tur6 (12): share turnover is the average of daily turnover over the
# past 6(12) months (require at least 50(100) days). Daily turnover is
# traded volumes divided by total number of shares.
#
# dvol6 (12): dollar trading volume is the average of daily dolloar
# trading volume over the past 6(12) months (require at least 50(100)
# days). Daily dollar trading volume is traded volumes times stock price.
# And finally, take the natural logarithm.
#
# illiq6 (12): illiquidity is the average of daily illiquidity over the
# past 6(12) months (require at least 50(100) days). Daily illiquidity
# is absolute value of return divided by daily dollar trading volume.
#
# Volume is adjusted for NASDAQ stocks based on Gao and Ritter (2010)
#
# Hou, Xue and Zhang (2020)
# "We adjust the NASDAQ trading volume to account for the institutional
# differences between NASDAQ and NYSE-Amex volumes (Gao and Ritter 2010).
# Prior to February 1, 2001, we divide NASDAQ volume by two.
# On February 1, 2001, a "riskless principal" rule goes into effect and
# results in a reduction of approximately 10% in reported volume. From
# February 1, 2001 to December 31, 2001, we divide NASDAQ volume by 1.8.
# During 2002, securities firms began to charge institutional investors
# commissions on NASDAQ trades, rather than the prior practice of marking
# up or down the net price. This practice reduces reported volume by
# roughly 10%. For 2002 and 2003, we divide NASDAQ volume by 1.6. For
# 2004 and later years, we use a divisor of one."
# ----------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import os
import time

class ap_volume:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP daily data
        # wrds package introduced new argument of `chunksize` from version 3.1.0
        # to reduce memory usage and avoid memory error when retrieving large
        # dataset. However, this requires dataframe appending which has poor
        # performance. To speed up data retrieving, I do not apply `chunksize`.
        # This will extract 77,734,734 (rows) by 7 (columns) and this is time
        # consuming: around 40 mins (it might take more than 2 times longer if
        # `chunksize` is enabled).
        # Make `chunksize` enabled if you get memory error.
        dsf = conn.raw_sql("""
            select a.permno, a.date, a.shrout, a.vol, a.ret, a.prc, b.exchcd
            from crsp.dsf a left join crsp.msenames b
                on a.permno=b.permno and a.date>=b.namedt and a.date<=b.nameendt
            where b.exchcd between -2 and 3 and b.shrcd between 10 and 11
        """, date_cols=['date'], chunksize=None)
        end_time = time.time()
        sql_time = (end_time-start_time) / 60
        print('\n--------- Extract data from WRDS ---------')
        print(f'Time used (SQL): {sql_time: 3.1f} mins')

        start_time = time.time()
        dsf = dsf.drop_duplicates(['permno', 'date'], keep='last')
        dsf['prc'] = dsf['prc'].abs()
        dsf.loc[dsf['vol']<0, 'vol'] = np.nan
        dsf.loc[dsf['shrout']<=0, 'shrout'] = np.nan
        dsf.loc[dsf['prc']<=0, 'prc'] = np.nan
        dsf.loc[dsf['ret']<=-1, 'ret'] = np.nan
        dsf['permno'] = dsf['permno'].astype(int)
        # Adjust volume for NASDAQ stocks
        mask1 = (dsf['date']<'2001-02-01') & (dsf['exchcd']==3)
        dsf.loc[mask1, 'vol'] = dsf['vol'] / 2
        mask2 = ((dsf['date']>='2001-02-01') & (dsf['date']<='2001-12-31')
            & (dsf['exchcd']==3))
        dsf.loc[mask2, 'vol'] = dsf['vol'] / 1.8
        mask3 =  ((dsf['date']>='2002-01-01') & (dsf['date']<='2003-12-31')
            & (dsf['exchcd']==3))
        dsf.loc[mask3, 'vol'] = dsf['vol'] / 1.6
        dsf['shrout'] = dsf['shrout'] * 1000
        dsf['dvol_d'] = dsf['vol'] * dsf['prc']
        dsf['to_d'] = dsf['vol'] / dsf['shrout']
        dsf.loc[dsf['dvol_d']>0, 'illiq_d'] = dsf['ret'].abs() / dsf['dvol_d']
        dsf = dsf.drop(columns=['shrout', 'prc', 'ret', 'vol', 'exchcd'])
        self.dsf = dsf.copy()

        end_time = time.time()
        print(f'Obs: {len(dsf)}')
        print(f'Time used (clean): {(end_time-start_time)/60: 3.1f} mins\n')

    def vol_est(self, j, min_n, var, var_name):
        start_time = time.time()
        df = self.dsf.copy()
        df['yyyymm'] = df['date'].dt.year * 100 + df['date'].dt.month

        to_msum = (df.groupby(['permno', 'yyyymm'])
            [var].sum(min_count=1).to_frame('var_m').reset_index())
        to_mcount = (df.groupby(['permno', 'yyyymm'])
            [var].count().to_frame('n').reset_index())
        df = to_msum.merge(to_mcount, how='inner', on=['permno', 'yyyymm'])
        df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m')
        df['date'] = df['date'] + pd.offsets.MonthEnd(0)
        df['midx'] = (df['date'].dt.year-1925) * 12 + df['date'].dt.month - 11
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        df['var_sum'] = (df.groupby('permno')['var_m']
            .rolling(window=j, min_periods=1).sum().reset_index(drop=True))
        df['day_sum'] = (df.groupby('permno')['n']
            .rolling(window=j, min_periods=1).sum().reset_index(drop=True))
        df['lmidx'] = df.groupby('permno')['midx'].shift(j-1)
        df.loc[df['day_sum']<=0, 'day_sum'] = np.nan
        df['v'] = df['var_sum'] / df['day_sum']
        df.loc[df['day_sum']<min_n, 'v'] = np.nan
        df['month_gap'] = df['midx'] - df['lmidx']
        # Control month gap
        df.loc[df['month_gap']!=j-1, 'v'] = np.nan
        df = df.query('v==v').copy()
        df[var_name] = df['v']
        df = df[['permno', 'yyyymm', var_name]]
        if var == 'dvol_d':
            df.loc[df[var_name]<=0, var_name] = np.nan
            df[var_name] = np.log(df[var_name])

        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'\n--------- {var_name} ---------')
        print(f'Obs: {len(df)}')
        print(f'Time used: {end_time-start_time: 3.1f} seconds\n')
        return df

if __name__ == '__main__':
    db = ap_volume()
    to6 = db.vol_est(6, 50, 'to_d', 'tur6')
    to12 = db.vol_est(12, 100, 'to_d', 'tur12')
    dvol6 = db.vol_est(6, 50, 'dvol_d', 'dvol6')
    dvol12 = db.vol_est(12, 100, 'dvol_d', 'dvol12')
    illiq6 = db.vol_est(6, 50, 'illiq_d', 'illiq6')
    illiq12 = db.vol_est(12, 100, 'illiq_d', 'illiq12')
    vol = (to6.merge(to12, how='outer', on=['permno', 'yyyymm'])
        .merge(dvol6, how='outer', on=['permno', 'yyyymm'])
        .merge(dvol12, how='outer', on=['permno', 'yyyymm'])
        .merge(illiq6, how='outer', on=['permno', 'yyyymm'])
        .merge(illiq12, how='outer', on=['permno', 'yyyymm']))
    print(f'Obs: {len(vol)}')
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    vol.to_csv(os.path.join(data_dir, 'volume.txt'), sep='\t', index=False)
    print('Done: data is generated')
