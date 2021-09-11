# ------------------------------------------------------------------
#                   Analysts' earnings forecasts
#
# cov: analyst coverage (number of analysts)
# disp: analyst forecasts dispersion. It is defined as standard
# deviation of forecasts (stdev) scaled by absolute value of mean
# forecasts (meanest). Keep if forecasts are made 30 days before
# forecast period end data. Require mean forecasts not equal to 0
# and at least 2 analysts.
#
# 0.39% of obs are zero mean forecasts
# 21.83% of obs made forecasts less than 30 days before forecast
# period end date (some are after forecast period end date)
#
# Available from 197601
#
# Hou, Xue and Zhang (2020)
#
# Use CRSP-IBES link table from WRDS to get PERMNO
# Require link score less than or equal to 2
# score=1: 8-digit historical CUSIP matching. IBES CUSIPs and the
# CRSP NCUSIP
# score=2: Historical ticker matching plus 6-digit CUSIPs and
# similar company names (spedis1 < 30)
# See details: https://wrds-www.wharton.upenn.edu/documents/796/IBES_CRSP_Linking_Table_by_WRDS.pdf
#
# Need to pay attention to extreme values of dispersion: might
# winsorize to kick out outliers.
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta

class ap_analysts:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP-IBES link table
        crsp_ibes_link = conn.raw_sql("""
            select ticker, permno, sdate, edate
            from wrdsapps.ibcrsphist
            where score<=2
        """, date_cols=['sdate', 'edate'])

        crsp_ibes_link['permno'] = crsp_ibes_link['permno'].astype(int)

        # Extract IBES unadjusted file
        ibes = conn.raw_sql("""
            select ticker, statpers, numest, meanest, stdev, fpedats
            from ibes.statsumu_epsus
            where fpi='1' and measure='EPS' and usfirm=1 and curcode='USD'
            order by ticker, statpers
        """, date_cols=['statpers', 'fpedats'])

        print('\n--------- Extract data from WRDS ---------')
        print(f'Obs (raw): {len(ibes)}')
        ibes = ibes.drop_duplicates(['ticker', 'statpers'], keep='last')
        print(f'Obs (after removing duplicates): {len(ibes)}')
        ibes = ibes.merge(crsp_ibes_link, how='inner', on='ticker')
        # Ensure valid link period
        ibes = ibes.query('sdate<=statpers<=edate').copy()
        ibes['yyyymm'] = ibes['statpers'].dt.year*100 + ibes['statpers'].dt.month
        obs = len(ibes)
        obs_0meanest = len(ibes.query('meanest==0'))
        print(f'Obs (with valid link): {obs}')
        print(f'Percent (zero meanest): {obs_0meanest/obs: 3.2%}')
        # Keep if the forecasts are made 30 days before the forecast period end
        ibes['date_gap'] = ibes['fpedats'] - ibes['statpers']
        obs_less30days = len(ibes[ibes['date_gap']<=timedelta(days=30)])
        print(f'Percent (forecast made less than 30 days): {obs_less30days/obs: 3.2%}')
        ibes = ibes[ibes['date_gap']>timedelta(days=30)].copy()
        ibes = ibes[['permno', 'yyyymm', 'numest', 'stdev', 'meanest']].copy()
        print(f'Obs (with valid forecasts): {len(ibes)}')
        self.ibes = ibes.copy()

        end_time = time.time()
        print(f'Time used: {end_time-start_time: 3.1f} seconds')

    def analysts_est(self):
        df = self.ibes.copy()
        # Require meanest not equal to 0
        df['disp'] = np.where(df['meanest']==0, np.nan,
            df['stdev'] / df['meanest'].abs())
        # Require at least 2 analysts
        df.loc[df['numest']<2, 'disp'] = np.nan
        df = df.rename(columns={'numest': 'cov'})
        df = df.drop(columns=['stdev', 'meanest'])
        df['cov'] = df['cov'].astype(int)
        df = df[['permno', 'yyyymm', 'cov', 'disp']]
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)
        return df

if __name__ == '__main__':
    db = ap_analysts()
    analysts = db.analysts_est()
    analysts = analysts.sort_values(['permno', 'yyyymm'], ignore_index=True)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    analysts.to_csv(os.path.join(data_dir, 'analysts.txt'),
        sep='\t', index=False)
    print('Done: data is generated')
