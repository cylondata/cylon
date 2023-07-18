import time
import argparse

import pandas as pd
from numpy.random import default_rng
from pycylon.frame import CylonEnv, DataFrame
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.dotdict import dotdict
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import writefile
from pycylon.net.ucc_config import UCCConfig
from pycylon.net.redis_ucc_oob_context import UCCRedisOOBContext
from pycylon.net.reduce_op import ReduceOp
import boto3
from botocore.exceptions import ClientError
import os

import logging
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def join(data=None):
    global ucc_config
    StopWatch.start(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    redis_context = UCCRedisOOBContext(data['world_size'], f"tcp://{data['redis_host']}:{data['redis_port']}")

    if redis_context is not None:
        ucc_config = UCCConfig(redis_context)

    if ucc_config is None:
        print("unable to initialize uccconfig")



    env = CylonEnv(config=ucc_config, distributed=True)

    context = env.context

    if context is None:
        print("unable to retrieve cylon context")

    communicator = context.get_communicator()

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    for i in range(data['it']):
        env.barrier()
        StopWatch.start(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")
        t1 = time.time()
        df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1) * 1000
        # sum_t = comm.reduce(t)
        sum_t = communicator.allreduce(t, ReduceOp.SUM)
        # tot_l = comm.reduce(len(df3))
        tot_l = communicator.allreduce(len(df3), ReduceOp.SUM)

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l, file=open(data['output_summary_filename'], 'a'))
            StopWatch.stop(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'], object_name=data['s3_stopwatch_object_name'])
        upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_summary_object_name'])

    env.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cylon scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    parser.add_argument('-i', dest='it', type=int, default=10)
    parser.add_argument('-u', dest='unique', type=float, default=0.9, help="unique factor")
    parser.add_argument('-s', dest='scaling', type=str, default='w', choices=['s', 'w'],
                        help="s=strong w=weak")
    parser.add_argument('-w', dest='world_size', type=int, help="world size", required=True)
    parser.add_argument("-r", dest='redis_host', type=str, help="redis address, default to 127.0.0.1",
                        default="127.0.0.1")
    parser.add_argument("-p1", dest='redis_port', type=int, help="name of redis port", default=6379)
    parser.add_argument("-p2", dest='ucx_port', type=int, help='ucx listen port', default=50002)
    parser.add_argument('-f1', dest='output_scaling_filename', type=str, help="Output filename for scaling results",
                        required=True)
    parser.add_argument('-f2', dest='output_summary_filename', type=str, help="Output filename for scaling summary results",
                        required=True)
    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", required=True)
    parser.add_argument('-o1', dest='s3_stopwatch_object_name', type=str, help="S3 Object Name", required=True)
    parser.add_argument('-o2', dest='s3_summary_object_name', type=str, help="S3 Object Name", required=True)

    args = vars(parser.parse_args())
    args['host'] = "aws"
    join(args)

    # os.system(f"{git} branch | fgrep '*' ")
    # os.system(f"{git} rev-parse HEAD")
