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
from math import ceil
from typing import List

import numpy as np
from pyarrow import Table as ArrowTable
from pyarrow import csv

from pytwisterx.util.FileUtils import files_exist
from pytwisterx.util.FileUtils import path_exists

'''
Supporting Data Loading from DL Workloads
References: https://github.com/pytorch/pytorch
'''



class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataLoader(object):

    def __init__(self, source_dir: str = None, source_files: List = [], source_file_names: List[str] = [],
                 file_type: str = 'csv', loader_type='arrow', delimiter=',', file_types: List[str] = [],
                 loader_types: List[str] = [], delimiters: List[str] = []):
        path_exists(path=source_dir)
        files_exist(dir_path=source_dir, files=source_files)
        self._source_dir = source_dir
        self._source_files = source_files
        self._source_file_names = source_file_names
        self._file_type = file_type
        self._loader_type = loader_type
        self._delimiter = delimiter
        self._file_types = file_types
        self._loader_types = loader_types
        self._delimiters = delimiters
        self._dataset: List = None

    @property
    def source_dir(self) -> str:
        return self._source_dir

    @property
    def source_files(self) -> List[str]:
        return self._source_files

    @property
    def source_file_names(self) -> List[str]:
        return self._source_file_names

    @property
    def file_type(self) -> str:
        return self._file_type

    @property
    def loader_type(self) -> str:
        return self._loader_type

    @property
    def delimiter(self) -> str:
        return self._delimiter

    @property
    def dataset(self) -> List:
        if not len(self._dataset) or self._dataset is None:
            raise RuntimeError("load method not called or something went wrong! {}".format(len(self._dataset)))
        return self._dataset

    @dataset.setter
    def dataset(self, values: List = []):
        self._dataset = values

    def load(self):
        raise NotImplementedError("Base class Not Implemented Method")


class LocalDataLoader(DataLoader):

    def load(self):
        print("Local Load")
        if self.loader_type is None:
            self.loader_type = 'arrow'
        if self.loader_type == 'arrow':
            _loaded_data_list: List[ArrowTable] = []
            for id, file in enumerate(self.source_files):
                fpath = os.path.join(self.source_dir, file)
                print("Loading File {}, {}".format(id, fpath))
                self.source_file_names.append('source_file_' + str(id))
                _loaded_data_list.append(csv.read_csv(fpath))
            self.dataset = _loaded_data_list
        elif self.loader_type == 'numpy':
            raise NotImplementedError("Numpy Loader is not implemented!")
        else:
            raise NotImplementedError("The Loader Type {} is not supported!".format(self.loader_type))


class DistributedDataLoader(DataLoader):
    pass


class MiniBatcher(object):

    @staticmethod
    def generate_minibatches(data=None, minibatch_size=1):
        """
        TODO: Write proper method definition
        This method executes the following
            Look total data and create mini-batches
            Each mini-batch has a user-defined batch size
            For instance 15,000 data divided for batch size 32
            There can 469 partitions.
            But we cannot use re-shape
            create 468 * 32 = 14976 ( 468 batches)
            and create the 469 batch with re-using already used values
            with the remaining values 15000-14976=24
            So adding 8 extra values from existing data
            do this randomly
            take a look at the code of Pytorch how they do it.
        :param data: input data
        :param minibatch_size: mini-batch size expected in training
        :return: mini-batched data set
        """
        data_shape = data.shape
        num_batches = ceil(data_shape[0] / float(minibatch_size))
        remainder = data_shape[0] % minibatch_size
        is_remainder = False
        init_additional_idx = np.arange(5)
        additional_records = data[init_additional_idx, :]
        records = None
        if remainder > 0:
            is_remainder = True
            additional_idx = np.arange(minibatch_size)
            record_idx = np.arange((num_batches - 1) * minibatch_size)
            additional_records = data[additional_idx, :]
            additional_records_shape = additional_records.shape
            additional_records = np.reshape(additional_records, (1, additional_records_shape[0],
                                                                 additional_records_shape[1]))
            records = data[record_idx, :]
            records = np.reshape(records, (num_batches - 1, minibatch_size, data_shape[1]))
            records = np.concatenate((records, additional_records), axis=0)
        else:
            records = np.reshape(data, (num_batches, minibatch_size, data_shape[1]))
        return records
