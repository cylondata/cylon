import subprocess
import time
import argparse

import boto3
from botocore.exceptions import ClientError
import os

import logging

def environ_or_required(key):
    return (
        {'default': os.environ.get(key)} if os.environ.get(key)
        else {'required': True}
    )

def get_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # download the file
    s3_client = boto3.client('s3')
    try:
        with open(file_name, 'wb') as f:
            s3_client.download_fileobj(bucket, object_name, f)
        return f
    except ClientError as e:
        logging.error(e)
        return None



def join(data=None):
    script = get_file(file_name=data['output_filename'], bucket=data['s3_bucket'], object_name=data['s3_object_name'])

    if script is None:
        print(f"unable to retrieve file {data['output_filename']} from AWS S3")

    cmd = data['args'].split()
    subprocess.call(['python'] + [data['output_filename']] + cmd, shell=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run S3 script")

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))
    parser.add_argument('-o', dest='s3_object_name', type=str, help="S3 Object Name", **environ_or_required('S3_OBJECT_NAME'))
    parser.add_argument('-f', dest='output_filename', type=str, help="Output filename",
                        **environ_or_required('OUTPUT_FILENAME'))
    parser.add_argument('-a', dest='args', type=str, help="script exec arguments",
                        **environ_or_required('EXEC_ARGS'))



    args = vars(parser.parse_args())
    join(args)
