# -------------------------------------------------------------------------
#                       WRDS: Apache Parquet
#
# Convert raw data from WRDS to parquet format to achieve fast I/O
#
# -------------------------------------------------------------------------


import pandas as pd
import os


class wrds_to_parquet():
    def __init__(self, infile, outfile):
        raw_dir = '/Users/ml/Data/wrds/raw'
        pq_dir = '/Users/ml/Data/wrds/parquet'
        self.infile = os.path.join(raw_dir, infile+'.txt.gz')
        self.outfile = os.path.join(pq_dir, outfile+'.parquet')

    def read_data(self):
        chunks = pd.read_csv(self.infile, sep='\t',
            chunksize=1e6, low_memory=False)
        df = pd.concat(chunks, ignore_index=True)
        df.columns = df.columns.str.lower()
        return df

    def data_summary(self, data, datevar):
        begdate = data[datevar].min()
        enddate = data[datevar].max()
        obs = len(data)
        print('--------------------')
        print('Date range: {0} -- {1}'.format(begdate, enddate))
        print('Obs: {0}'.format(obs))
        print('--------------------')

    # ----------  CRSP  ---------------
    def crsp(self):
        df = self.read_data()
        convert_list = ['siccd', 'hsiccd', 'dlretx', 'dlret', 'ret', 'retx']
        for i in convert_list:
            df[i] = pd.to_numeric(df[i], errors='coerce')

        df.to_parquet(self.outfile)
        self.data_summary(df, 'date')

    # --------  Compustat North America fundamentals  --------
    def compf(self):
        df = self.read_data()
        df.to_parquet(self.outfile)
        self.data_summary(df, 'datadate')


wrds_to_parquet('msf_1925_2019', 'msf').crsp()

wrds_to_parquet('dsf_1925_2000','dsf1').crsp()
wrds_to_parquet('dsf_2001_2019','dsf2').crsp()

wrds_to_parquet('funda_1950_2019','funda').compf()

wrds_to_parquet('fundq_1961_2019','fundq').compf()


