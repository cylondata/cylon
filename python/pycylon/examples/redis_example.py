from pycylon.net.ucc_config import UCCConfig
from pycylon.net.redis_ucc_oob_context import UCCRedisOOBContext
from pycylon import Table, CylonEnv
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import argparse

if __name__ == "__main__":
    print("Initializing redis...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', "-n", type=int, help="world size")
    parser.add_argument("--redis_host", "-r", type=str, help="redis address, default to 127.0.0.1",
                        default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, help="name of redis port", default=6379)
    args = parser.parse_args()
    redis_context = UCCRedisOOBContext(args.world_size, f"tcp://{args.redis_host}:{args.port}")

    if redis_context is not None:
        ucc_config = UCCConfig(redis_context)

        if ucc_config is None:
            print("unable to initialize uccconfig")

        env = CylonEnv(config=ucc_config)

        print("cylon env initialized")

        csv_read_options = CSVReadOptions() \
            .use_threads(False) \
            .block_size(1 << 30) \
            .na_values(['na', 'none'])

        rank = env.rank + 1

        print(f"rank: {rank}")
        csv1 = f"/home/qad5gv/cylon/data/input/user_usage_tm_{rank}.csv"
        csv2 = f"/home/qad5gv/cylon/data/input/user_device_tm_{rank}.csv"

        first_table: Table = read_csv(env, csv1, csv_read_options)
        second_table: Table = read_csv(env, csv2, csv_read_options)

        print(first_table)

        first_row_count = first_table.row_count

        second_row_count = second_table.row_count

        print(f"Table 1 & 2 Rows [{first_row_count},{second_row_count}], "
              f"Columns [{first_table.column_count},{second_table.column_count}]")

        configs = {'join_type': 'inner', 'algorithm': 'sort'}
        joined_table: Table = first_table.distributed_join(table=second_table,
                                                           join_type=configs['join_type'],
                                                           algorithm=configs['algorithm'],
                                                           left_on=[3],
                                                           right_on=[0]
                                                           )

        join_row_count = joined_table.row_count

        print(f"First table had : {first_row_count} and Second table had : {second_row_count}, "
              f"Joined has : {join_row_count}")

        env.finalize()






    else:
        print("unable to initialize redis oob context")

