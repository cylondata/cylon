#!/bin/sh
. /opt/conda/etc/profile.d/conda.sh
conda activate cylon_dev
export LD_LIBRARY_PATH=/cylon/install/lib
echo $LD_LIBRARY_PATH
python /cylon/aws/scripts/S3_run_script.py