from pycylon.data.table import Table
from typing import List
from numbers import Number
import warnings


class Index(object):

    def __init__(self, data):
        pass

    def initialize(self):
        pass

    @property
    def index(self):
        return self


class NumericIndex(Index):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, data):
        self._index_values = data
        self.initialize()

    def initialize(self):
        pass

    @property
    def index_values(self):
        return self._index_values

    @index_values.setter
    def index_values(self, data):
        self._index_values = data


class IntegerIndex(NumericIndex):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, data):
        super().__init__(data=data)

    def initialize(self):
        super().initialize()


class RangeIndex(IntegerIndex):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, data=None, start: int = 0, stop: int = 0, step: int = 0):
        self._start = start
        self._stop = stop
        self._step = step
        super().__init__(data=data)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start

    @property
    def stop(self):
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step

    def initialize(self):
        if self.stop == 0:
            if isinstance(self.data, range):
                self._index_values = self.data
                self.start = self.data.start
                self.stop = self.data.stop
                self.step = self.data.step
            else:
                warnings.warn("Empty Range!. Range data or range criteria must be passed")
        else:
            self._index_values = range(self.start, self.stop, self.step)
