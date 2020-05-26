import os
import sys
from typing import List


def path_exists(path: str = None):
    if path is None:
        raise ValueError("Directory path is None")
    return os.path.exists(path)


def files_exist(dir_path: str = None, files: List = []):
    dir_exists = path_exists(path=dir_path)
    if dir_exists:
        if len(files):
            for file in files:
                fpath = os.path.join(dir_path, file)
                if not path_exists(path=fpath):
                    raise ValueError("File {} doesn't exist in the given fileset".format(fpath))
