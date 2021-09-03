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
#include "test_utils.hpp"
#include <cylon/util/sort.hpp>
#include <random>

TEST_CASE("Testing introsort", "[util]") {
  SECTION("testing introsort") {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    int elements = 10000;
    std::uniform_int_distribution<uint64_t> distrib(0, elements);
    int64_t *index = new int64_t[elements];
    int64_t *value = new int64_t[elements];
    for (int i = 0; i < elements; i++) {
      index[i] = i;
      value[i] = distrib(gen);
    }
    cylon::util::introsort(value, index, elements);
    bool sorted = true;
    for (int i = 0; i < elements - 1; i++) {
      if (value[i] > value[i + 1]) {
        sorted = false;
      }
    }
    REQUIRE(sorted == true);
  }

  SECTION("testing introsort") {
    int elements = 1000000;
    int64_t *index = new int64_t[elements];
    int64_t *value = new int64_t[elements];
    for (int i = 0; i < elements; i++) {
      index[i] = i;
      value[i] = i;
    }
    cylon::util::introsort(value, index, elements);
    bool sorted = true;
    for (int i = 0; i < elements - 1; i++) {
      if (value[i] > value[i + 1]) {
        sorted = false;
      }
    }
    REQUIRE(sorted == true);
  }
}