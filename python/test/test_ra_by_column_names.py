from pycylon.csv import csv_reader
from pycylon import Table
from pycylon import CylonContext
import argparse

parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')
parser.add_argument('--table2_path', type=str, help='Path to table 2 csv')

args = parser.parse_args()


def single_process(args):
    ctx: CylonContext = CylonContext(config=None)

    tb1: Table = csv_reader.read(ctx, args.table1_path, ',')

    tb2: Table = csv_reader.read(ctx, args.table2_path, ',')

    print(tb1.column_names)
    print(tb2.column_names)

    configs = {'join_type': 'inner', 'algorithm': 'sort', 'left_col': 0,
               'right_col': 0}

    tb3: Table = tb1.join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=[0],
                          right_on=[0]
                          )

    tb3.show()

    tb4: Table = tb1.join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=['A'],
                          right_on=['A']
                          )

    tb4.show()

    tb4: Table = tb1.join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          on=['A']
                          )

    tb4.show()

    tb5: Table = tb1.join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          on=[0]
                          )

    tb5.show()

    ctx.finalize()


def multi_process(args):
    ctx: CylonContext = CylonContext(config='mpi')

    tb1: Table = csv_reader.read(ctx, args.table1_path, ',')

    tb2: Table = csv_reader.read(ctx, args.table2_path, ',')

    print(tb1.column_names)
    print(tb2.column_names)

    configs = {'join_type': 'inner', 'algorithm': 'sort', 'left_col': 0,
               'right_col': 0}

    tb3: Table = tb1.distributed_join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=[0],
                          right_on=[0]
                          )

    tb3.show()

    tb4: Table = tb1.distributed_join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=['A'],
                          right_on=['A']
                          )

    tb4.show()

    tb4: Table = tb1.distributed_join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          on=['A']
                          )

    tb4.show()

    tb5: Table = tb1.distributed_join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          on=[0]
                          )

    tb5.show()

    ctx.finalize()

single_process(args)

#multi_process(args)