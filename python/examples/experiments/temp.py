import argparse
import logging
import os
import time

import pyarrow
from pyarrow import csv as ar_csv
from pycylon.ctx.context import CylonContext
from pycylon.data.table import Table
from pycylon.data.table import csv_reader



ctx: CylonContext = CylonContext("mpi")

rank: int = ctx.get_rank()
world_size: int = ctx.get_world_size()

print(" sdasdas", world_size)

ctx.finalize()

