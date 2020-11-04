import os
import time
import download_util
from pycylon import CylonContext
from pycylon import Table
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import numpy as np
import pandas as pd


def load_aggregated_single_response_pandas(target='AUC', min_r2_fit=0.3, max_ec50_se=3.0,
                                           combo_format=False,
                                           rename=True):
    url = "https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_single_response_agg"
    output_combined_single_response = \
        "/home/vibhatha/data/uno/Pilot1/workload_1/combined_single_response_agg"

    if not os.path.exists(output_combined_single_response):
        download_util.download(url=url, output_file=output_combined_single_response)

    if os.path.exists(output_combined_single_response):
        print(f"Pandas Data file : {output_combined_single_response}")
        t1 = time.time()
        df = pd.read_csv(output_combined_single_response, engine='c', sep='\t',
                         dtype={'SOURCE': str, 'CELL': str, 'DRUG': str, 'STUDY': str,
                                'AUC': np.float32, 'IC50': np.float32,
                                'EC50': np.float32, 'EC50se': np.float32,
                                'R2fit': np.float32, 'Einf': np.float32,
                                'HS': np.float32, 'AAC1': np.float32,
                                'AUC1': np.float32, 'DSS1': np.float32})
        t2 = time.time()
        # df = df[(df['R2fit'] >= min_r2_fit) & (df['EC50se'] <= max_ec50_se)]
        # filter_time = time.time() - t2
        # print("Pandas Data Loading Time ", df.shape, t2 - t1)
        # print("Pandas Filter Time 1", df.shape, filter_time)

        t8 = time.time()
        df['R2fit'] >= min_r2_fit
        t9 = time.time()
        print("Pandas Comparison + Extract Time ", (t9 - t8), df.shape)


def load_aggregated_single_response_cylon(target='AUC', min_r2_fit=0.3, max_ec50_se=3.0,
                                          combo_format=False,
                                          rename=True):
    url = "https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_single_response_agg"
    output_combined_single_response = \
        "/home/vibhatha/data/uno/Pilot1/workload_1/combined_single_response_agg"

    if not os.path.exists(output_combined_single_response):
        download_util.download(url=url, output_file=output_combined_single_response)

    if os.path.exists(output_combined_single_response):
        print(f"Data file : {output_combined_single_response}")
        ctx: CylonContext = CylonContext(config=None, distributed=False)
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t")
        t1 = time.time()
        tb: Table = read_csv(ctx, output_combined_single_response, csv_read_options)
        t2 = time.time()
        # tb = tb[(tb['R2fit'] >= min_r2_fit) & (tb['EC50se'] <= max_ec50_se)]
        # t3 = time.time()
        # table_read_time = t2 - t1
        # filter_time = t3 - t2
        # print("Cylon ", tb.row_count, tb.column_count, tb.column_names)
        # print("Cylon Data Loading Time: ", table_read_time)
        # print("Cylon Data Filter Time: ", filter_time)
        # t4 = time.time()
        # npy = tb.to_numpy(zero_copy_only=False)
        # t5 = time.time()

        # print("Npy Conv time", t5 - t4)

        t6 = time.time()
        npyc = tb['R2fit'].to_arrow().combine_chunks().columns[0].chunks[0].to_numpy()
        t7 = time.time()
        print(t7 - t6, npyc.dtype, npyc.shape[0])
        t8 = time.time()
        tbx = tb['R2fit'] >= min_r2_fit
        t9 = time.time()
        print("Cylon Comparison + Extract Time ", (t9 - t8), tbx.row_count)


# load_aggregated_single_response_pandas()
#load_aggregated_single_response_cylon()

import numpy as np
from pycylon.data.compute import comparison_compute_op

ar = np.random.random(10_000_000)

import inspect
import operator

spec = inspect.getfullargspec(comparison_compute_op)
print(ar.shape)
t1 = time.time()
val = comparison_compute_op(ar, 0.5, operator.__gt__)
t2 = time.time()
print("Cython Time :", t2 - t1)
list1 = []
t1 = time.time()
for i in range(ar.shape[0]):
    list1.append(operator.__gt__(ar[i], 0.5))
t2 = time.time()
print("Python Time :", t2 - t1)

t1 = time.time()
a = ar[ar > 0.5]
t2 = time.time()
print("Numpy Time :", t2 - t1)