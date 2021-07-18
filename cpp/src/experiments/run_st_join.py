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

import os
from os.path import expanduser

from cpp.src.experiments.generate_csv import generate_file

home = expanduser("~")
# join_exec = f"{home}/git/cylon/build/bin/table_api_test_hash"
join_exec = f"../../../build/bin/table_join_st_test"
print(f"twx home: {join_exec}", flush=True)

csvs = ["/tmp/csv1.csv", "/tmp/csv2.csv"]

out_dir = f"/tmp/twx_join_test/"
print(f"output dir: {out_dir}", flush=True)
os.system(f"rm -rf {out_dir}; mkdir -p {out_dir}")

cols = 4
key_duplication_ratio = 0.99  # on avg there will be rows/key_range_ratio num of duplicate keys

repetitions = 10
print("repetitions for each test", repetitions, flush=True)

for i in [int(ii * 1000000) for ii in [0.1, 0.25, 0.5, 0.75, 1, 10, 25, 50, 75, 100, 250, 500]]:
    print(f"\n\n##### test {i} starting!", flush=True)

    test_dir = f"{out_dir}/{i}"
    os.system(f"rm -rf {test_dir}; mkdir -p {test_dir}")

    krange = (0, int(i * key_duplication_ratio))
    for f in csvs:
        generate_file(output=f, rows=i, cols=cols, krange=krange)

    for r in range(repetitions):
        os.system(f"{join_exec}")
        print(f"\n\n##### {r+1}/{repetitions} iter done!")

# os.system(f"mv {csv1} {test_dir}")
    # os.system(f"mv {csv2} {test_dir}")
    # for j in ['right', 'left', 'inner', 'outer']:
    #     os.system(f"mv /tmp/h_out_{j}.csv {test_dir}/")
    #     os.system(f"mv /tmp/s_out_{j}.csv {test_dir}/")

    print(f"\n\n##### test {i} done!\n-----------------------------------------", flush=True)
