import os


import time
import argparse
import math
import subprocess
import os 
import gc
import sys

import time
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext, SQLContext

TOTAL_NODES = 10


if __name__ == "__main__":
    w = int(sys.argv[1])
    input_dir = sys.argv[2]
    spark_master = sys.argv[3]
    it = int(sys.argv[4])
    
    
    procs = int(math.ceil(w / TOTAL_NODES))
    print("procs per worker", procs, " iter ", it, flush=True)

    assert procs <= 16

    spark = SparkSession.builder \
                        .appName(f"Spark join {w} {r}") \
                        .config("spark.master", spark_master) \
                        .config("spark.executor.memory", "10g") \
                        .config("spark.executor.cores", 1) \
                        .config("spark.memory.fraction", 0.75) \
                        .getOrCreate()

    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

    df_l = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
                .load(inputDir + "/csv1_*").repartition(w).cache()
    print("#### spark left " + df_l.count())
    
    df_r = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
                .load(inputDir + "/csv2_*").repartition(w).cache()
    print("#### spark right " + df_r.count())

    for i in range(it):
        t1 = time.time()
        out = df_l.join(df_r, on=['0'], how='inner')
        t2 = time.time()
        
        lines = q.count()

        print(f"##### pyark {i + 1}/{it} iter start! SPLIT_FROM_HERE", flush=True)
        print(f"###time {w} {i} {(t2 - t1) * 1000:.0f}, {lines}", flush=True)

#         del df_l 
#         del df_r
#         del out 
#         gc.collect()

