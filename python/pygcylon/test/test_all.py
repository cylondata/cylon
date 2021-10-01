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
import numpy as np

print("-------------------------------------------------")
print("|\t\tPyGcylon Test Framework\t\t|")
print("-------------------------------------------------")

responses = []


def test_pygcylon_installation():
    print("1. PyGcylon Installation Test")
    responses.append(os.system("pytest -q python/pygcylon/test/test_pygcylon.py"))
    assert responses[-1] == 0


def test_cylon_context():
    print("2. CylonContext Test")
    responses.append(
        os.system(
            "mpirun --oversubscribe --allow-run-as-root -n 2 python -m pytest --with-mpi "
            "-q python/pycylon/test/test_cylon_context.py"))
    assert responses[-1] == 0


def test_shuffle():
    print("3. PyGcylon shuffle Test")
    responses.append(
        os.system(
            "mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi "
            "-q python/pygcylon/test/test_shuffle.py"))
    assert responses[-1] == 0


def test_setops():
    print("4. PyGcylon Set Operations Test")
    responses.append(
        os.system(
            "mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi "
            "-q python/pygcylon/test/test_setops.py"))
    assert responses[-1] == 0


def test_groupby():
    print("5. PyGcylon GroupBy Test")
    responses.append(
        os.system(
            "mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi "
            "-q python/pygcylon/test/test_groupby.py"))
    assert responses[-1] == 0


def test_join():
    print("6. PyGcylon Join Test")
    responses.append(
        os.system(
            "mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi "
            "-q python/pygcylon/test/test_join.py"))
    assert responses[-1] == 0


def test_sort():
    print("7. PyGcylon sort Test")
    responses.append(
        os.system(
            "mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi "
            "-q python/pygcylon/test/test_sort.py"))
    assert responses[-1] == 0


def test_all():
    ar = np.array(responses)
    total = len(responses)
    failed_count = sum(ar > 0)

    if failed_count > 0:
        print(f"{failed_count} of {total}  Tests Failed !!!")
        assert False
    else:
        print("All Tests Passed!")
