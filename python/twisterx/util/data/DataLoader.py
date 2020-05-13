import os
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import List
from pytwisterx.utils.file import path_exists
from pytwisterx.utils.file import files_exist
from pyarrow import csv
from pyarrow import Table as ArrowTable


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
    def dataset(self, values:List = []):
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
