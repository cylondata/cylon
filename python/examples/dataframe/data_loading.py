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

from pycylon import DataFrame, read_csv, CylonEnv
from pycylon.net import MPIConfig
import sys
import pandas as pd


# using cylon native reader
df = read_csv(sys.argv[1])
print(df)

# using pandas to load csv
df = DataFrame(pd.read_csv(
    "http://data.un.org/_Docs/SYB/CSV/SYB63_1_202009_Population,%20Surface%20Area%20and%20Density.csv", skiprows=6, usecols=[0, 1]))
print(df)

# loading json
df = DataFrame(pd.read_json("https://api.exchangerate-api.com/v4/latest/USD"))
print(df)

# distributed loading : run in distributed mode with MPI or UCX
env = CylonEnv(config=MPIConfig())
df = read_csv(sys.argv[1], slice=True, env=env)
print(df)
