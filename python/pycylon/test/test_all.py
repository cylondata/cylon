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
print("|\t\tPyCylon Test Framework\t\t|")
print("-------------------------------------------------")

responses = []

def get_mpi_command():
    if os.name == 'posix':
        return "mpirun --oversubscribe --allow-run-as-root"
    elif os.name == 'nt':
        return "mpiexec"
    else:
        return ""

def test_pycylon_installation_test():
    print("1. PyCylon Installation Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_pycylon.py"))
    assert responses[-1] == 0


def test_pyarrow_installation_test():
    print("2. PyArrow Installation Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_build_arrow.py"))
    assert responses[-1] == 0


# def test_fake():
#     # NOTE: To Test the Test Framework
#     print("Fake Test")
#     responses.append(os.system("pytest -q python/pycylon/test/test_fake.py"))
#     assert responses[-1] == 0

def test_cylon_context():
    print("3. CylonContext Test")
    responses.append(
        os.system(
            get_mpi_command() + " -n 2 python -m pytest --with-mpi "
            "-q python/pycylon/test/test_cylon_context.py"))
    assert responses[-1] == 0


def test_channel():
    print("4. Channel Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_channel.py"))
    assert responses[-1] == 0


def test_commtype():
    print("5. CommType Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_comm_type.py"))
    assert responses[-1] == 0


def test_csv_read_options():
    print("6. CSV Read Options Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_csv_read_options.py"))
    assert responses[-1] == 0


def test_datatype():
    print("7. Data Types Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_data_types.py"))
    assert responses[-1] == 0


def test_data_utils():
    print("8. Data Utils Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_data_utils.py"))
    assert responses[-1] == 0


def test_status():
    print("9. Cylon Status Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_status.py"))
    assert responses[-1] == 0


def test_request():
    print("10. Request Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_txrequest.py"))
    assert responses[-1] == 0


def test_pycylon_pyarrow():
    print("11. PyArrow/PyCylon Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_pyarrow.py"))
    assert responses[-1] == 0


def test_table_conversion():
    print("12. Table Conversion Test")
    responses.append(os.system(
        get_mpi_command() + " -n 2 python -m pytest --with-mpi "
        "-q python/pycylon/test/test_cylon_table_conversion.py"))
    assert responses[-1] == 0


def test_table_operation():
    print("13. Table Operation Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_table.py"))
    assert responses[-1] == 0


def test_table_properties():
    print("14. Table Properties Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_table_properties.py"))
    assert responses[-1] == 0


def test_aggregate():
    print("15. Aggregate Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_aggregate.py"))
    assert responses[-1] == 0


def test_join_config():
    print("16. Join Config Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_join_config.py"))
    assert responses[-1] == 0


def test_simple_table_join():
    print("17. Simple Table Join Test")
    responses.append(os.system(
        get_mpi_command() + " -n 4 python -m pytest --with-mpi "
        "-q python/pycylon/test/test_cylon_simple_table_join.py"))
    assert responses[-1] == 0


def test_dist_rl():
    print("18. Distributed Relational Algebra Operator Test")
    responses.append(
        os.system(
            get_mpi_command() + " -n 4 python -m pytest --with-mpi "
            "-q python/pycylon/test/test_dist_rl.py"))
    assert responses[-1] == 0


def test_rl():
    print("19. Sequential Relational Algebra Operator Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_rl.py"))
    assert responses[-1] == 0


def test_rl_col():
    print("20. Sequential Relational Algebra with Column Names Test")
    responses.append(os.system("pytest -q python/pycylon/test/test_ra_by_column_names.py"))
    assert responses[-1] == 0


def test_dist_rl_col():
    print("21. Distributed Relational Algebra with Column Names Test")
    responses.append(
        os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi -q "
                  "python/pycylon/test/test_dist_ra_by_column_names.py"))
    assert responses[-1] == 0


def test_index():
    print("22. Index Test")
    responses.append(
        os.system("pytest -q python/pycylon/test/test_index.py"))
    assert responses[-1] == 0


def test_compute():
    print("23. Compute Test")
    responses.append(
        os.system("pytest -q python/pycylon/test/test_compute.py"))
    assert responses[-1] == 0


def test_series():
    print("24. Series Test")
    responses.append(
        os.system("pytest -q python/pycylon/test/test_series.py"))
    assert responses[-1] == 0


def test_frame():
    print("25. DataFrame Test")
    responses.append(
        os.system("pytest -q python/pycylon/test/test_frame.py"))
    assert responses[-1] == 0


def test_duplicate():
    print("26. Duplicate Handling")
    responses.append(
        os.system(
            get_mpi_command() + " -n 2 python -m pytest --with-mpi "
            "-q python/pycylon/test/test_duplicate_handle.py"))
    assert responses[-1] == 0


def test_sorting():
    print("27. Sorting")
    responses.append(os.system("pytest -q python/pycylon/test/test_sorting.py"))
    assert responses[-1] == 0


def test_df_dist_sorting():
    print("28. Sorting")
    responses.append(os.system(get_mpi_command() + " -n 4 python -m pytest "
                               "-q python/pycylon/test/test_df_dist_sorting.py"))
    assert responses[-1] == 0

    
def test_pd_read_csv():
    print("29. pandas read_csv")
    responses.append(os.system("pytest -q python/pycylon/test/test_pd_read_csv.py"))
    assert responses[-1] == 0

def test_data_split():
    print("30. Data Split")
    responses.append(os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi python/pycylon/test/test_data_split.py"))
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
