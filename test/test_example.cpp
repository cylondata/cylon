//=====================================
// Sample test ...
//=====================================

#include "common/test_header.hpp"
#include <mpi.h>
TEST_CASE("sample_test_fixture", "[sample_group]") {

  SECTION("addition") {
    MPI_Init(NULL, NULL);
    REQUIRE((5+2) > 2);

    MPI_Finalize();
  }

  SECTION("substraction") {
    MPI_Init(NULL, NULL);
    REQUIRE((5-2)==3);

    MPI_Finalize();
  }
}


