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
