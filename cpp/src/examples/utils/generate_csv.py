import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='generate random data')
parser.add_argument('-o', dest='output', type=str, help='output file', default='/tmp/csv.csv')
parser.add_argument('-r', dest='rows', type=int, help='number of rows', default=10)
parser.add_argument('-c', dest='cols', type=int, help='number of cols', default=4)
parser.add_argument('-k', dest='idx_cols', type=int, nargs='+', help='index columns', default=[0])
parser.add_argument('--krange', nargs=2, type=int, help='key range', default=(0, 10))
parser.add_argument('--vrange', nargs=2, type=float, help='val range', default=(0., 1.))
parser.add_argument('--no_header', action='store_true', help='exclude header')

args = parser.parse_args()
args = vars(args)

print("generate csv :", args)

out_file = args['output']
rows = args['rows']
cols = args['cols']
idx_cols = args['idx_cols']
vrange = args['vrange']
krange = args['krange']
no_header = args['no_header']

df = pd.DataFrame(np.random.rand(rows, cols) * (vrange[1] - vrange[0]) + vrange[0], columns=list(range(cols)))

for i in idx_cols:
    assert cols > i >= 0
    df[i] = df[i].map(lambda x: int(krange[0] + (x - vrange[0]) * (krange[1] - krange[0]) / (vrange[1] - vrange[0])))

df.to_csv(out_file, header=not no_header, index=False)

pass
