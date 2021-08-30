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

from pycylon import Table
from pycylon import CylonContext
import pandas as pd
import numpy as np
import time

ctx = CylonContext(config=None, distributed=False)

records = 1_000_000
values = []
for col in range(0, 1):
    values.append(np.random.random(records))
pdf_t = pd.DataFrame(values)


tb_t = Table.from_pandas(ctx, pdf_t)

t1 = time.time()
pdf_t.to_dict()
t2 = time.time()
tb_t.to_pydict(with_index=True)
t3 = time.time()

print(f"Pandas Dictionary Conversion Time {t2-t1} s")
print(f"PyCylon Dictionary Conversion Time {t3-t2} s")