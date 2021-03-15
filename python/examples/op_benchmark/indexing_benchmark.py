import time
import pandas as pd
import numpy as np
import pycylon as cn
from pycylon import Table
from pycylon.io import CSVWriteOptions
from pycylon.indexing.index import IndexingSchema

ctx = cn.CylonContext(config=None, distributed=False)


def generate_data():
    dataset = []
    cols = ['a', 'b', 'c', 'd']
    records = 10_000_000
    duplicate_factor = 0.9
    gen_record_size = int(records * duplicate_factor)

    for col in cols:
        record = np.random.randint(gen_record_size, size=records)
        dataset.append(record)



    tb = cn.Table.from_numpy(ctx, cols, dataset)
    pdf = tb.to_pandas()

    csv_write_options = CSVWriteOptions().with_delimiter(',')
    tb.to_csv(f'/tmp/indexing_{records}_{duplicate_factor}.csv', csv_write_options)


def do_indexing():
    index_file = "/tmp/indexing_10000000_0.9.csv"

    pdf = pd.read_csv(index_file)
    ct = Table.from_pandas(ctx, pdf)

    # NOTE: make sure this value exist in the generated data file
    search_value = 3458646

    t0 = time.time()
    ct.set_index('a', IndexingSchema.LINEAR)
    t1 = time.time()
    pdf1 = pdf.set_index('a')
    t2 = time.time()
    df1 = pdf1.loc[search_value, 'b' : 'd']
    t3 = time.time()
    tb1 = ct.loc[search_value, 'b': 'd']
    t4 = time.time()
    cn_indexing_time = t1-t0
    pd_indexing_time = t2-t1
    pd_loc_time = t3-t2
    cn_loc_time = t4-t3
    print(f"Indexing speed up : {pd_indexing_time/ cn_indexing_time}")
    print(f"Loc speed up : {pd_loc_time / cn_loc_time}")

    #print(df1)


def check_indexing_validity():
    pdf = pd.read_csv('/tmp/duplicate_data_0.csv')
    pdf = pdf.set_index('a')

    print(pdf)

    df1 = pdf.loc[4:1]
    print(df1)

#generate_data()
#do_indexing()
#check_indexing_validity()