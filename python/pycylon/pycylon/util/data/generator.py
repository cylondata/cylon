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


import numpy as np
import logging


def generate_numeric_csv(rows: int, columns: int, file_path: str):
    """
    this method generates a csv file with numeric data with a given datatype, columns and rows
    :param rows: number of rows in the csv file
    :param columns: number of columns in the csv file
    :param dtype: data type
    """
    a: np.ndarray = np.random.random(rows * columns)
    np.random.random_sample(100)
    a.reshape(shape=(rows, columns))
    np.savetxt(file_path, a, delimiter=',')
    logging.info("Numeric CSV saved %s", file_path)
