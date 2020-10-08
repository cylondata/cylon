from pycylon.csv import csv_reader
from pycylon import Table
from pycylon import CylonContext
import argparse

table1_path = '/tmp/user_device_tm_1.csv'
table2_path = '/tmp/user_usage_tm_1.csv'


def single_process():
    ctx: CylonContext = CylonContext(config=None)

    tb1: Table = csv_reader.read(ctx, table1_path, ',')

    tb2: Table = csv_reader.read(ctx, table2_path, ',')

    print(tb1.column_names)
    print(tb2.column_names)

    configs = {'join_type': 'inner', 'algorithm': 'sort'}

    tb3: Table = tb1.join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=[0],
                          right_on=[3]
                          )

    tb3.show()

    tb4: Table = tb1.join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=['use_id'],
                          right_on=['use_id']
                          )

    tb4.show()

    tb4: Table = tb1.join(ctx, table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          on=['use_id']
                          )

    tb4.show()

    # tb5: Table = tb1.join(ctx, table=tb2,
    #                       join_type=configs['join_type'],
    #                       algorithm=configs['algorithm'],
    #                       on=[0]
    #                       )
    #
    # tb5.show()

    ctx.finalize()


def multi_process():
    ctx: CylonContext = CylonContext(config='mpi')

    tb1: Table = csv_reader.read(ctx, table1_path, ',')

    tb2: Table = csv_reader.read(ctx, table2_path, ',')

    print(tb1.column_names)
    print(tb2.column_names)

    configs = {'join_type': 'inner', 'algorithm': 'sort', 'left_col': 0,
               'right_col': 0}

    tb3: Table = tb1.distributed_join(ctx, table=tb2,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      left_on=[0],
                                      right_on=[3]
                                      )

    tb3.show()

    tb4: Table = tb1.distributed_join(ctx, table=tb2,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      left_on=['use_id'],
                                      right_on=['use_id']
                                      )

    tb4.show()

    tb4: Table = tb1.distributed_join(ctx, table=tb2,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      on=['use_id']
                                      )

    tb4.show()

    # tb5: Table = tb1.distributed_join(ctx, table=tb2,
    #                       join_type=configs['join_type'],
    #                       algorithm=configs['algorithm'],
    #                       on=[0]
    #                       )
    #
    # tb5.show()

    ctx.finalize()


single_process()

multi_process()
