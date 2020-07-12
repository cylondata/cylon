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


import time


def time_conversion(time: int, time_type: str = "ms") -> int:
    tx: int = time
    if time_type is None:
        raise ValueError("Time Type cannot be None")
    else:
        if time_type == "ms":
            tx = tx / 1000000.0
        elif time_type == "us":
            tx = tx / 1000.0
        elif time_type == "s":
            tx = tx / 1000000000.0
        elif time_type == "ns":
            pass
    return tx


def benchmark_with_repitions(repititions: int = 10, time_type: str = "ms"):
    def wrap(f):
        def wrapped_f(*args):
            t1 = time.time_ns()
            for i in range(repititions):
                rets = f(*args)
            t2 = time.time_ns()
            tx = time_conversion(time=t2 - t1, time_type=time_type)
            return (tx) / float(repititions), rets
        return wrapped_f

    return wrap
