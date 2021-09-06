# ------------------------------------------------------------------
#                 Standardised unexpected earnings
#
# sue: it is the ratio of earnings change scaled by the standard
# deviation of the changes over past 8 quarters (require at least 6
# quarters). Earnings change is the difference between current
# earnings and earnings 4 quarters before.
# Earning is split adjusted: eps = epspxq / ajexq

# sur: it is the ratio of revenue per share change scaled by the
# standard deviation of the changes over past 8 quarters (require at
# least 6 quarters). Revenue per share change is the difference
# between current revenue per share and revenue per share 4 quarters
# before. Here uses saleq instead of revtq since saleq has nearly
# 100,000 more observations than revtq
# Revenue per share is split adjusted: rps = saleq / (cshprq*ajexq)

# Hou, Xue and Zhang (2020)
# Chen and Zimmermann (2021)
#
# Number of firms that have epspxq and saleq is small before fiscal
# year 1962
# Start from fiscal year 1962 when retrieving data from Compustat
#
# To get PERMNO for each GVKEY, NCUSIP-CUSIP map is applied
# It will get more matched pairs if CCM access is available
#
# Example
#  gvkey    datadate    fyearq   fqtr
# 123456   2010-03-31    2010     1
# Distribute earnings info to monthly data
#  gvkey    datadate    fyearq   fqtr  yyyymm
# 123456   2010-03-31    2010     1    201006
# 123456   2010-03-31    2010     1    201007
# 123456   2010-03-31    2010     1    201008
# From Jun to Aug in 2010, earnings for fiscal quarter 2010Q1
# will be used (assume earnings are available with 3-month lag)
#
# Further control to avoid data errors and forward-looking bias
# RDQ is the first public report date for earnings information
# RDQ < datadate: 0.04% (release date is before fiscal quarter
# end, these might be erroneous records)
# RDQ >= datadate + 90 days: 3.05% (earnings are not available
# with 3-month lag)
# eps is set to missing for the above two secnarios
#
# Observations of SUE is too small before 196409, so data is
# kept from 196409
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil import relativedelta
import os
import time
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class ap_sue:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP daily data
        # Start from fiscal year 1962
        fundq = conn.raw_sql("""
            select gvkey, datadate, fyearq, fqtr, rdq,
                epspxq, saleq, cshprq, ajexq
            from comp.fundq
            where consol='C' and popsrc='D' and datafmt='STD'
                and curcdq='USD' and indfmt='INDL' and fyearq>=1962
                and fqtr is not null
        """, date_cols=['datadate', 'rdq'])

        # Keep the most recent one for each fiscal quarter
        fundq = fundq.sort_values(['gvkey', 'fyearq', 'fqtr', 'datadate'],
            ignore_index=True)
        fundq = fundq.drop_duplicates(['gvkey', 'fyearq', 'fqtr'], keep='last')
        # split-adjusted EPS
        fundq.loc[fundq['ajexq']<=0, 'ajexq'] = np.nan
        fundq['eps'] = fundq['epspxq'] / fundq['ajexq']
        fundq['rps'] = fundq['saleq'] / (fundq['cshprq']*fundq['ajexq'])
        fundq['gap'] = fundq['rdq'] - fundq['datadate']
        obs0 = len(fundq[fundq['gap']<timedelta(days=0)])
        obs90 = len(fundq[fundq['gap']>timedelta(days=90)])
        # Set to missing if report date is before datadate
        fundq.loc[fundq['gap']<timedelta(days=0), 'eps'] = np.nan
        fundq.loc[fundq['gap']<timedelta(days=0), 'rps'] = np.nan
        # Assume information is available with 90 days (3 months) lag
        # Set to missing if report date is outside 90 days lag
        fundq.loc[fundq['gap']>timedelta(days=90), 'eps'] = np.nan
        fundq.loc[fundq['gap']>timedelta(days=90), 'rps'] = np.nan
        fundq['date'] = fundq['datadate'] + pd.offsets.MonthEnd(0)
        fundq['date'] = fundq['date'] + pd.offsets.MonthEnd(3)
        # Generate quarter index to control quarter gaps later
        qidx = fundq[['fyearq', 'fqtr']].copy()
        qidx = qidx.drop_duplicates(['fyearq', 'fqtr']).copy()
        qidx = qidx.sort_values(['fyearq', 'fqtr'], ignore_index=True)
        qidx['qidx'] = qidx.index + 1

        fundq = fundq.merge(qidx, how='left', on=['fyearq', 'fqtr'])
        fundq = fundq[['gvkey', 'date', 'eps', 'rps', 'qidx', 'datadate']]
        fundq = fundq.sort_values(['gvkey', 'datadate'], ignore_index=True)
        self.fundq = fundq.copy()

       # PERMNO-GVKEY link for common shares in NYSE/AMEX/NASDAQ
        permno_gvkey = conn.raw_sql("""
            select distinct a.permno, b.gvkey, c.namedt, c.nameendt
            from crsp.msenames a
            inner join comp.security b on a.ncusip=substring(b.cusip, 1, 8)
            inner join crsp.msenames c on a.permno=c.permno
            where a.shrcd between 10 and 11 and a.exchcd between -2 and 3
                and b.excntry='USA' and a.ncusip is not null
                and b.cusip is not null
            order by permno, gvkey, namedt
        """, date_cols=['namedt', 'nameendt'])

        permno_gvkey['permno'] = permno_gvkey['permno'].astype(int)
        self.permno_gvkey = permno_gvkey.copy()

        end_time = time.time()
        print('\n--------- Extract data from WRDS ---------')
        print(f'Percent (rdq<datadate): {obs0/len(fundq): 3.2%}')
        print(f'Percent (rdq>datadate+90): {obs90/len(fundq): 3.2%}')
        print(f'time_used: {end_time-start_time: 3.1f} seconds\n')

    def sue_est(self, var, name):
        start_time = time.time()
        df = self.fundq.copy()

        df = df.sort_values(['gvkey', 'qidx'], ignore_index=True)
        df['l4'+var] = df.groupby(['gvkey'])[var].shift(4)
        df['l4qidx'] = df.groupby(['gvkey'])['qidx'].shift(4)
        df['qgap'] = df['qidx'] - df['l4qidx']
        # 4 quarters lag info is set to missing if the quarter gap is not 4
        df.loc[df['qgap']!=4, 'l4'+var] = np.nan
        df[var+'_diff'] = df[var] - df['l4'+var]
        df = df.sort_values(['gvkey', 'qidx'], ignore_index=True)
        # Use difference in past 8 quarters and require at least 6 quarters
        df[var+'_diff_std'] = (df.groupby('gvkey')[var+'_diff']
            .rolling(window=8, min_periods=6).std().reset_index(drop=True))
        df['l7qidx'] = df.groupby(['gvkey'])['qidx'].shift(7)
        df['qgap'] = df['qidx'] - df['l7qidx']
        df.loc[df[var+'_diff_std']<=0, var+'_diff_std'] = np.nan
        # Set to missing if outside past 8 quarters
        df.loc[df['qgap']!=7, var+'_diff_std'] = np.nan
        df[name] = df[var+'_diff'] / df[var+'_diff_std']
        df = df[['gvkey', 'date', 'datadate', name]].copy()

        # Expand data to distibute surprise to monthly frequence
        df = pd.concat([df]*3, ignore_index=True)
        df['month_gap'] = (df.groupby(['gvkey', 'date'])
            ['date'].cumcount())
        df = df.sort_values(['gvkey', 'date', 'month_gap'], ignore_index=True)
        df['month_gap'] = (df['month_gap']
            .apply(lambda x: relativedelta.relativedelta(months=x)))
        df['date'] = df['date'] + df['month_gap'] + pd.offsets.MonthEnd(0)
        del df['month_gap']
        df = df.sort_values(['gvkey', 'date', 'datadate'], ignore_index=True)
        df = df.drop_duplicates(['gvkey', 'date'], keep='last').copy()
        df['yyyymm'] = df['date'].dt.year*100 + df['date'].dt.month
        # Get PERMNO to SUE data with date range condition
        # TODO: use CRSP-Compustat Merged data if available
        df = df.merge(self.permno_gvkey, how='left', on='gvkey')
        df = df.query('namedt<=datadate<=nameendt').copy()
        df = df[['permno', 'yyyymm', name, 'gvkey', 'datadate']]
        df = df.sort_values(['permno', 'yyyymm', 'datadate'], ignore_index=True)
        df = df.drop_duplicates(['permno', 'yyyymm'], keep='last').copy()
        df['permno'] = df['permno'].astype('int')
        # Obs are too small before 196409
        df = df[(df[name].notna()) & (df['yyyymm']>=196409)].copy()
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- {name.upper()} estimation ---------')
        print(f'Obs: {len(df)}')
        print(f'time_used: {(end_time-start_time)/60: 3.1f} mins')
        return df

if __name__ == '__main__':
    db = ap_sue()
    sue = db.sue_est('eps', 'sue')
    sur = db.sue_est('rps', 'sur')
    sue = sue.merge(sur, how='outer', on=['permno', 'yyyymm'])
    sue = sue.sort_values(['permno', 'yyyymm'], ignore_index=True)
    obs = len(sue)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    sue.to_csv(os.path.join(data_dir, 'sue.txt'), sep='\t', index=False)
    print('Done: data is generated')
    print(f'Obs: {obs}')
