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

print("-------------------------------------------------")
print("|\t\tPyCylon Test Framework\t\t|")
print("-------------------------------------------------")


def get_mpi_command():
    if os.name == 'posix':
        return "mpirun --oversubscribe --allow-run-as-root"
    elif os.name == 'nt':
        return "mpiexec"
    else:
        return ""


def assert_success(fn):
    def wrapper():
        assert fn() == 0

    return wrapper


@assert_success
def test_pycylon_installation_test():
    print("1. PyCylon Installation Test")
    return os.system("pytest -q python/pycylon/test/test_pycylon.py")


@assert_success
def test_pyarrow_installation_test():
    print("2. PyArrow Installation Test")
    return os.system("pytest -q python/pycylon/test/test_build_arrow.py")


@assert_success
def test_cylon_context():
    print("3. CylonContext Test")
    return os.system(
        get_mpi_command() + " -n 2 python -m pytest --with-mpi "
                            "-q python/pycylon/test/test_cylon_context.py")


@assert_success
def test_channel():
    print("4. Channel Test")
    return os.system("pytest -q python/pycylon/test/test_channel.py")


@assert_success
def test_commtype():
    print("5. CommType Test")
    return os.system("pytest -q python/pycylon/test/test_comm_type.py")


@assert_success
def test_csv_read_options():
    print("6. CSV Read Options Test")
    return os.system("pytest -q python/pycylon/test/test_csv_read_options.py")


@assert_success
def test_datatype():
    print("7. Data Types Test")
    return os.system("pytest -q python/pycylon/test/test_data_types.py")


@assert_success
def test_data_utils():
    print("8. Data Utils Test")
    return os.system("pytest -q python/pycylon/test/test_data_utils.py")


@assert_success
def test_status():
    print("9. Cylon Status Test")
    return os.system("pytest -q python/pycylon/test/test_status.py")


@assert_success
def test_request():
    print("10. Request Test")
    return os.system("pytest -q python/pycylon/test/test_txrequest.py")


@assert_success
def test_pycylon_pyarrow():
    print("11. PyArrow/PyCylon Test")
    return os.system("pytest -q python/pycylon/test/test_pyarrow.py")


@assert_success
def test_table_conversion():
    print("12. Table Conversion Test")
    return os.system(
        get_mpi_command() + " -n 2 python -m pytest --with-mpi "
                            "-q python/pycylon/test/test_cylon_table_conversion.py")


@assert_success
def test_table_operation():
    print("13. Table Operation Test")
    return os.system("pytest -q python/pycylon/test/test_table.py")


@assert_success
def test_table_properties():
    print("14. Table Properties Test")
    return os.system("pytest -q python/pycylon/test/test_table_properties.py")


@assert_success
def test_aggregate():
    print("15. Aggregate Test")
    return os.system("pytest -q python/pycylon/test/test_aggregate.py")


@assert_success
def test_join_config():
    print("16. Join Config Test")
    return os.system("pytest -q python/pycylon/test/test_join_config.py")


@assert_success
def test_simple_table_join():
    print("17. Simple Table Join Test")
    return os.system(
        get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                            "-q python/pycylon/test/test_cylon_simple_table_join.py")


@assert_success
def test_dist_rl():
    print("18. Distributed Relational Algebra Operator Test")
    return os.system(
        get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                            "-q python/pycylon/test/test_dist_rl.py")


@assert_success
def test_rl():
    print("19. Sequential Relational Algebra Operator Test")
    return os.system("pytest -q python/pycylon/test/test_rl.py")


@assert_success
def test_rl_col():
    print("20. Sequential Relational Algebra with Column Names Test")
    return os.system("pytest -q python/pycylon/test/test_ra_by_column_names.py")


@assert_success
def test_dist_rl_col():
    print("21. Distributed Relational Algebra with Column Names Test")
    return os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi -q "
                                         "python/pycylon/test/test_dist_ra_by_column_names.py")


@assert_success
def test_index():
    print("22. Index Test")
    return os.system("pytest -q python/pycylon/test/test_index.py")


@assert_success
def test_compute():
    print("23. Compute Test")
    return os.system("pytest -q python/pycylon/test/test_compute.py")


@assert_success
def test_series():
    print("24. Series Test")
    return os.system("pytest -q python/pycylon/test/test_series.py")


@assert_success
def test_frame():
    print("25. DataFrame Test")
    return os.system("pytest -q python/pycylon/test/test_frame.py")


@assert_success
def test_duplicate():
    print("26. Duplicate Handling")
    return os.system(
        get_mpi_command() + " -n 2 python -m pytest --with-mpi "
                            "-q python/pycylon/test/test_duplicate_handle.py")


@assert_success
def test_sorting():
    print("27. Sorting")
    return os.system("pytest -q python/pycylon/test/test_sorting.py")


@assert_success
def test_df_dist_sorting():
    print("28. Sorting")
    return os.system(get_mpi_command() + " -n 4 python -m pytest "
                                         "-q python/pycylon/test/test_df_dist_sorting.py")


@assert_success
def test_pd_read_csv():
    print("29. pandas read_csv")
    return os.system("pytest -q python/pycylon/test/test_pd_read_csv.py")


@assert_success
def test_data_split():
    print("30. Data Split")
    return os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                                         "python/pycylon/test/test_data_split.py")


@assert_success
def test_repartition():
    print("31. Repartition")
    return os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                                         "python/pycylon/test/test_repartition.py")


@assert_success
def test_equals():
    print("32. Equals")
    return os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                                         "python/pycylon/test/test_equal.py")


@assert_success
def test_parquet():
    print("33. DataFrame Test")
    return os.system("pytest -q python/pycylon/test/test_parquet.py")


@assert_success
def test_dist_aggregate():
    print("34. Dist Aggregates")
    return os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                                         "python/pycylon/test/test_dist_aggregate.py")


@assert_success
def test_dist_io():
    print("34. Dist IO")
    return os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                                         "python/pycylon/test/test_io.py")


@assert_success
def test_custom_mpi_comm():
    print("36. Custom mpi comm")
    return os.system(
        f"{get_mpi_command()} -n 4 pytest python/pycylon/test/test_custom_mpi_comm.py ")


if os.environ.get('CYLON_GLOO'):
    @assert_success
    def test_gloo():
        print("35. Gloo")
        return os.system("python -m pytest python/pycylon/test/test_gloo.py")


    @assert_success
    def test_gloo_mpi():
        print("36. Gloo")
        return os.system(
            f"{get_mpi_command()} -n 4 python -m pytest python/pycylon/test/test_gloo_mpi.py")


    @assert_success
    def test_custom_mpi_comm_gloo():
        print("36. Gloo custom mpi comm")
        return os.system(
            f"{get_mpi_command()} -n 4 pytest python/pycylon/test/test_custom_mpi_comm.py "
            f"--comm gloo-mpi")

if os.environ.get('CYLON_UCC'):
    @assert_success
    def test_ucx_mpi():
        print("37. UCX MPI")
        return os.system(
            f"{get_mpi_command()} -n 4 python -m pytest python/pycylon/test/test_ucx_mpi.py")


    @assert_success
    def test_custom_mpi_comm_ucx():
        print("36. UCX custom mpi comm")
        return os.system(
            f"{get_mpi_command()} -n 4 pytest python/pycylon/test/test_custom_mpi_comm.py "
            f"--comm ucx")

if os.environ.get('CYLON_GLOO') and os.environ.get('CYLON_UCC'):
    @assert_success
    def test_mpi_multiple_env_init():
        print("38. Create and destroy multiple environments in MPI")
        return os.system(
            f"{get_mpi_command()} -n 4 python -m pytest "
            f"python/pycylon/test/test_mpi_multiple_env_init.py")


@assert_success
def test_dist_slice():
    print("39. Dist Slice Test")
    return os.system(get_mpi_command() + " -n 4 python -m pytest --with-mpi "
                                         "python/pycylon/test/test_slice.py")
