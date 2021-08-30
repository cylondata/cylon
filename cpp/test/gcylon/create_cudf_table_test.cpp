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
  const int COLS = 2;
  const int ROWS = 10;
  const int START = 0;
  const bool CONT = false;

  SECTION("testing create table") {

    std::shared_ptr<cudf::table> tbl = constructTable(COLS, ROWS, START, CONT);
    auto tv = tbl->view();

    REQUIRE((tv.num_columns() == COLS && tv.num_rows() == ROWS));

    int64_t * col0 = getColumnPart<int64_t>(tv.column(0), START, ROWS);

    for (int i = 0; i < ROWS; i++) {
      REQUIRE((col0[i] == START + i));
    }
  }
}