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
