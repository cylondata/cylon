/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/test_header.hpp"
#include <gcylon/utils/util.hpp>
#include <gcylon/utils/construct.hpp>

using namespace cylon;
using namespace gcylon;

TEST_CASE("create cudf table testing", "[gcreate]") {
  cylon::Status status;
  const int COLS = 4;
  const int ROWS = 10;
  const int64_t START = 100;
  const int STEP = 5;
  const bool CONT = true;

  SECTION("testing create increasing table") {

    std::unique_ptr<cudf::table> tbl = constructTable(COLS, ROWS, START, STEP, CONT);
    auto tv = tbl->view();

    REQUIRE((tv.num_columns() == COLS && tv.num_rows() == ROWS));

    int64_t value = START;
    for (int j = 0; j < COLS; j++) {

      int64_t *col = getColumnPart<int64_t>(tv.column(j), 0, ROWS);
      if (!CONT)
        value = START;

      for (int i = 0; i < ROWS; i++) {
        REQUIRE((col[i] == value));
        value += STEP;
      }
    }
  }

  SECTION("testing create random table") {

    std::unique_ptr<cudf::table> tbl = constructRandomDataTable(COLS, ROWS);
    auto tv = tbl->view();

    REQUIRE((tv.num_columns() == COLS && tv.num_rows() == ROWS));
  }

}