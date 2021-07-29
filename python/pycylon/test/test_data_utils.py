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

'''
Run test:
>> pytest -q python/test/test_data_utils.py
'''

import os
from pycylon.util.data.DataManager import LocalDataLoader
import pyarrow


def test_data_loading():
    base_path = '/tmp'

    train_file_name: str = "user_usage_tm_1.csv"
    test_file_name: str = "user_device_tm_1.csv"

    assert os.path.exists(os.path.join(base_path, train_file_name))
    assert os.path.exists(os.path.join(base_path, test_file_name))

    dl = LocalDataLoader(source_dir=base_path, source_files=[train_file_name])

    print(dl.source_dir, dl.source_files, dl.file_type, dl.delimiter, dl.loader_type)

    dl.load()

    for id, dataset in enumerate(dl.dataset):
        assert isinstance(dataset, pyarrow.Table)
        assert dataset.to_pandas().to_numpy().shape == (240, 4)
