import time
import pandas as pd
import numpy as np
import pycylon as cn
from pycylon import Table
from pycylon.io import CSVWriteOptions


def generate_data():
    dataset = []
    cols = ['a', 'b', 'c', 'd']
    records = 10_000_000
    duplicate_factor = 0.9
    gen_record_size = int(records * duplicate_factor)

    for col in cols:
        record = np.random.randint(gen_record_size, size=records)
        dataset.append(record)

    ctx = cn.CylonContext(config=None, distributed=False)

    tb = cn.Table.from_numpy(ctx, cols, dataset)
    pdf = tb.to_pandas()

    csv_write_options = CSVWriteOptions().with_delimiter(',')
    tb.to_csv(f'/tmp/indexing_{records}_{duplicate_factor}.csv', csv_write_options)


def do_indexing():
    index_file = "/tmp/indexing_10000000_0.9.csv"

    pdf = pd.read_csv(index_file)

    print(pdf.shape)

    t1 = time.time()
    pdf1 = pdf.set_index('a')
    print(type(pdf1.index))
    t2 = time.time()
    df1 = pdf1.loc[237250, 'b' : 'd']
    t3 = time.time()
    print(t2-t1, t3-t2, df1.shape)

    #print(df1)


#generate_data()
do_indexing()