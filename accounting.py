# ------------------------------------------------------------------
#                           Accounting
#
# Hou, Xue and Zhang (2020)
# See appendix
# ------------------------------------------------------------------

import wrds
import configparser as cp
import pandas as pd
import numpy as np
from dateutil import relativedelta
import time
import os
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class ap_accounting:
    def __init__(self):
        start_time = time.time()
        pass_dir = '~/.pass'
        cfg = cp.ConfigParser()
        cfg.read(os.path.join(os.path.expanduser(pass_dir), 'credentials.cfg'))
        conn = wrds.Connection(wrds_username=cfg['wrds']['username'])

        # Extract CRSP daily data
        funda = conn.raw_sql("""
            select gvkey, datadate, fyear, cusip, at, ceq, pstk, capx, sale,
                invt, ppegt, che, dlc, dltt, mib, ppent, intan, ao, lo, dp,
                csho, ajex, act, lct, txp, ni, oancf, ivao, lt, ivst, ivncf,
                fincf, prstkc, sstk, dv
            from comp.funda
            where consol='C' and popsrc ='D' and datafmt = 'STD'
                and curcd = 'USD' and indfmt = 'INDL'
        """, date_cols=['datadate'])

        funda = funda.sort_values(['gvkey', 'fyear', 'datadate'],
            ignore_index=True)
        funda = funda.drop_duplicates(['gvkey', 'fyear'], keep='last')
        for i in ['at', 'ajex', 'csho']:
            funda.loc[funda[i]<=0, i] = np.nan

        for i in ['capx', 'sale', 'invt']:
            funda.loc[funda[i]<0, i] = np.nan

        for i in ['dlc', 'dltt', 'mib', 'pstk', 'sstk', 'prstkc', 'dv']:
            funda.loc[funda[i].isna(), i] = 0

        funda['date'] = funda['datadate'] + pd.offsets.MonthEnd(0)
        funda['date'] = funda['date'] + pd.offsets.MonthEnd(6)
        self.funda = funda.copy()

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
        print(f'Time used: {end_time-start_time: 3.1f} seconds\n')

    def accounting_est(self):
        start_time = time.time()
        df = self.funda.copy()

        # Abnormal corporate investment (aci)
        df.loc[(df['sale']>0) & (df['capx']>0), 'ce'] = df['capx'] / df['sale']
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        for i in range(1, 4):
            df['l'+str(i)+'ce'] = df.groupby('gvkey')['ce'].shift(i)

        df['l3fyear'] = df.groupby('gvkey')['fyear'].shift(3)
        df['fyear_gap'] = df['fyear'] - df['l3fyear']
        df['ce_avg3yr'] = (df['l1ce']+df['l2ce']+df['l3ce']) / 3
        df['aci'] = df['ce'] / df['ce_avg3yr'] - 1
        df.loc[df['fyear_gap']!=3, 'aci'] = np.nan
        df.loc[df['sale']<10, 'aci'] = np.nan
        df = (df.drop(columns=['ce', 'l1ce', 'l2ce', 'l3ce', 'ce_avg3yr',
            'l3fyear', 'fyear_gap']).copy())

        # Asset growth (ag)
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        df['l1at'] = df.groupby('gvkey')['at'].shift(1)
        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df['ag'] = df['at'] / df['l1at'] - 1
        df.loc[df['fyear_gap']!=1, 'ag'] = np.nan
        df = df.drop(columns=['l1at', 'l1fyear', 'fyear_gap']).copy()

        # Changes in PPE and inventory to assets (dpia)
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        for i in ['ppegt', 'invt', 'at']:
            df['l1'+i] = df.groupby('gvkey')[i].shift(1)
            df['d'+i] = df[i] - df['l1'+i]

        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df['dpia'] = (df['dppegt']+df['dinvt']) / df['l1at']
        df.loc[df['fyear_gap']!=1, 'dpia'] = np.nan
        df = (df.drop(columns=['l1at', 'dat', 'l1ppegt', 'dppegt', 'l1invt',
            'dinvt', 'l1fyear', 'fyear_gap']).copy())

        # Net operating assets (noa)
        # Changes in net operating assets (dnoa)
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        df['l1at'] = df.groupby('gvkey')['at'].shift(1)
        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df['oa'] = df['at'] - df['che']
        df['ol'] = (df['at'] - df['dlc'] - df['dltt'] - df['mib']
            - df['pstk'] - df['ceq'])
        df['noasset'] = df['oa'] - df['ol']
        df['noa'] = df['noasset'] / df['l1at']
        df.loc[df['fyear_gap']!=1, 'noa'] = np.nan

        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        df['l1noasset'] = df.groupby('gvkey')['noasset'].shift(1)
        df['dnoa'] = (df['noasset']-df['l1noasset']) / df['l1at']
        df.loc[df['fyear_gap']!=1, 'dnoa'] = np.nan
        df = (df.drop(columns=['l1at', 'l1fyear', 'fyear_gap', 'oa', 'ol',
            'noasset', 'l1noasset']).copy())

        # Changes in long-term net operating assets (dlno)
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        for i in ['ppent', 'intan', 'ao', 'lo', 'at']:
            df['l1'+i] = df.groupby('gvkey')[i].shift(1)
            df['d'+i] = df[i] - df['l1'+i]

        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df['dlnoasset'] = (df['dppent'] + df['dintan'] + df['dao']
            - df['dlo'] + df['dp'])
        df['at_avg2yr'] = (df['at']+df['l1at']) / 2
        df['dlno'] = df['dlnoasset'] / df['at_avg2yr']
        df.loc[df['fyear_gap']!=1, 'dlno'] = np.nan
        df = (df.drop(columns=['l1at', 'l1fyear', 'fyear_gap', 'l1ppent',
            'l1intan', 'l1ao', 'l1lo', 'dppent', 'dintan', 'dao', 'dlo',
            'dat', 'dlnoasset', 'at_avg2yr']).copy())

        # Investment growth (ig)
        # 2-year investment growth (ig2)
        # 3-year investment growth (ig3)
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        for i in range(1, 4):
            df['l'+str(i)+'capx'] = df.groupby('gvkey')['capx'].shift(i)
            df['l'+str(i)+'fyear'] = df.groupby('gvkey')['fyear'].shift(i)

        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df['fyear_gap2'] = df['fyear'] - df['l2fyear']
        df['fyear_gap3'] = df['fyear'] - df['l3fyear']
        df.loc[df['l1capx']!=0, 'ig'] = df['capx'] / df['l1capx'] - 1
        df.loc[df['l2capx']!=0, 'ig2'] = df['capx'] / df['l2capx'] - 1
        df.loc[df['l3capx']!=0, 'ig3'] = df['capx'] / df['l3capx'] - 1
        df.loc[df['fyear_gap']!=1, 'ig'] = np.nan
        df.loc[df['fyear_gap2']!=2, 'ig2'] = np.nan
        df.loc[df['fyear_gap3']!=3, 'ig3'] = np.nan
        df = (df.drop(columns=['l1capx', 'l1fyear', 'fyear_gap', 'l2capx',
            'l2fyear', 'fyear_gap2', 'l3capx', 'l3fyear', 'fyear_gap3']).copy())

        # Net stock issues (nsi)
        # Hou, Xue and Zhang (2020)
        # "we sort stocks with negative Nsi into two portfolios (1 and 2),
        # stocks with zero Nsi into 1 portfolio (3), and stocks with positive Nsi
        # into seven portfolios (4 to 10)"
        df['csho_adj'] = df['csho'] * df['ajex']
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        df['l1csho_adj'] = df.groupby('gvkey')['csho_adj'].shift(1)
        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df['nsi'] = np.log(df['csho_adj']/df['l1csho_adj'])
        df.loc[df['fyear_gap']!=1, 'nsi'] = np.nan
        df = (df.drop(columns=['csho_adj', 'l1csho_adj', 'l1fyear',
            'fyear_gap']).copy())

        # Composite debt issuance (cdi)
        df['bvdebt'] = df['dlc'] + df['dltt']
        df.loc[df['bvdebt']<=0, 'bvdebt'] = np.nan
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        df['l5bvdebt'] = df.groupby('gvkey')['bvdebt'].shift(5)
        df['l5fyear'] = df.groupby('gvkey')['fyear'].shift(5)
        df['fyear_gap'] = df['fyear'] - df['l5fyear']
        df['cdi'] = np.log(df['bvdebt']/df['l5bvdebt'])
        df.loc[df['fyear_gap']!=5, 'cdi'] = np.nan
        df = (df.drop(columns=['bvdebt', 'l5bvdebt', 'l5fyear',
            'fyear_gap']).copy())

        # Inventory growth (ivg)
        # Inventory changes (ivc)
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        df['l1invt'] = df.groupby('gvkey')['invt'].shift(1)
        df['l1at'] = df.groupby('gvkey')['at'].shift(1)
        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df.loc[df['l1invt']<=0, 'l1invt'] = np.nan
        df['ivg'] = df['invt'] / df['l1invt'] - 1
        df['at_avg2yr'] = (df['at']+df['l1at']) / 2
        df['ivc'] = (df['invt']-df['l1invt']) / df['at_avg2yr']
        df.loc[df['fyear_gap']!=1, 'ivg'] = np.nan
        df.loc[df['fyear_gap']!=1, 'ivc'] = np.nan
        df = (df.drop(columns=['l1invt', 'l1fyear', 'fyear_gap', 'l1at',
            'at_avg2yr']).copy())

        # Operating accruals (oa)
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        for i in ['act', 'che', 'lct', 'dlc', 'txp']:
            df['l1'+i] = df.groupby('gvkey')[i].shift(1)
            df['d'+i] = df[i] - df['l1'+i]

        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['l1at'] = df.groupby('gvkey')['at'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df.loc[df['dtxp'].isna(), 'dtxp'] = 0
        df.loc[df['fyear']<=1987, 'oa'] = ((df['dact']-df['dche'])
            - (df['dlct']-df['ddlc']-df['dtxp']) - df['dp'])
        df.loc[df['fyear']>1987, 'oa'] = df['ni'] - df['oancf']
        df['oa'] = df['oa'] / df['l1at']
        df.loc[df['fyear_gap']!=1, 'oa'] = np.nan
        df = (df.drop(columns=['l1act', 'l1che', 'l1lct', 'l1dlc', 'l1txp',
            'dact', 'dche', 'dlct', 'ddlc', 'dtxp', 'l1fyear', 'l1at',
            'fyear_gap']).copy())

        # Total accruals (oa)
        df['coa'] = df['act'] - df['che']
        df['col'] = df['lct'] - df['dlc']
        df['wc'] = df['coa'] - df['col']
        df['nca'] = df['at'] - df['act'] - df['ivao']
        df['ncl'] = df['lt'] - df['lct'] - df['dltt']
        df['nco'] = df['nca'] - df['ncl']
        df['fna'] = df['ivst'] + df['ivao']
        df['fnl'] = df['dltt'] + df['dlc'] + df['pstk']
        df['fin'] = df['fna'] - df['fnl']
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)
        for i in ['wc', 'nco', 'fin']:
            df['l1'+i] = df.groupby('gvkey')[i].shift(1)
            df['d'+i] = df[i] - df['l1'+i]

        df['l1fyear'] = df.groupby('gvkey')['fyear'].shift(1)
        df['l1at'] = df.groupby('gvkey')['at'].shift(1)
        df['fyear_gap'] = df['fyear'] - df['l1fyear']
        df.loc[df['fyear']<=1987, 'ta'] = df['dwc'] + df['dnco'] + df['dfin']
        df.loc[df['fyear']>1987, 'ta'] = (df['ni'] - df['oancf'] - df['ivncf']
            - df['fincf'] + df['sstk'] - df['prstkc'] - df['dv'])
        df['ta'] = df['ta'] / df['l1at']
        df.loc[df['fyear_gap']!=1, 'ta'] = np.nan

        # clean
        var_list = ['aci', 'ag', 'dpia', 'noa', 'dnoa', 'dlno', 'ig', 'ig2',
            'ig3', 'nsi', 'cdi', 'ivg', 'ivc', 'oa', 'ta']
        df = df[['gvkey', 'date', 'datadate', 'fyear']+var_list].copy()
        df = df.sort_values(['gvkey', 'fyear'], ignore_index=True)

        # Expand data to distibute surprise to monthly frequence
        df = pd.concat([df]*12, ignore_index=True)
        df['month_gap'] = df.groupby(['gvkey', 'date'])['date'].cumcount()
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
        df = df[['permno', 'yyyymm']+var_list+['gvkey', 'datadate']]
        df = df.sort_values(['permno', 'yyyymm', 'datadate'], ignore_index=True)
        df = df.drop_duplicates(['permno', 'yyyymm'], keep='last').copy()
        df = df.dropna(subset=var_list, how='all')
        df['permno'] = df['permno'].astype('int')
        df = df.sort_values(['permno', 'yyyymm'], ignore_index=True)

        end_time = time.time()
        print(f'--------- Accounting estimation ---------')
        print(f'Obs: {len(df)}')
        print(f'Time used: {(end_time-start_time)/60: 3.1f} mins')
        return df

if __name__ == '__main__':
    db = ap_accounting()
    acct = db.accounting_est()
    acct = acct.sort_values(['permno', 'yyyymm'], ignore_index=True)
    data_dir = '/Volumes/Seagate/asset_pricing_data'
    acct.to_csv(os.path.join(data_dir, 'accounting.txt'), sep='\t', index=False)
    print('Done: data is generated')
