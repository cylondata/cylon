##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

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


def generate_file(output='/tmp/csv.csv', rows=10, cols=4, idx_cols=None, vrange=(0., 1.),
                  krange=(0, 10), no_header=False):
    if idx_cols is None:
        idx_cols = [0]

    df = pd.DataFrame(np.random.rand(rows, cols) * (vrange[1] - vrange[0]) + vrange[0],
                      columns=list(range(cols)))

    for i in idx_cols:
        assert cols > i >= 0
        df[i] = df[i].map(lambda x: int(
            krange[0] + (x - vrange[0]) * (krange[1] - krange[0]) / (vrange[1] - vrange[0])))

    df.to_csv(output, header=not no_header, index=False, float_format='%.3f')


if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)

    print("generate csv :", args, flush=True)
    generate_file(**args)
