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

#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.h>
#include <chrono>

using namespace twisterx;
using namespace twisterx::join::config;

//template <const char* jtype>
bool RunJoin(const JoinConfig &jc,
             const std::shared_ptr<Table> &table1,
             const std::shared_ptr<Table> &table2,
             std::shared_ptr<Table> &output,
             const string &h_out_path) {
  Status status;

  auto t1 = std::chrono::high_resolution_clock::now();
  status = table1->Join(table2, jc, &output);
  auto t2 = std::chrono::high_resolution_clock::now();

  if (!status.is_ok()) {
    LOG(ERROR) << "Join failed!";
    return false;
  }
//  else {
//    status = output->WriteCSV(h_out_path);
//  }

  auto t3 = std::chrono::high_resolution_clock::now();

  output->Clear();

  if (status.is_ok()) {
    LOG(INFO) << "join_ms " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " write_ms "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    return true;
  } else {
    LOG(ERROR) << "Join write failed!";
    return false;
  }
}

int main(int argc, char *argv[]) {

  std::shared_ptr<Table> table1, table2, joined;
  Status status;

  LOG(INFO) << "Reading tables";
  auto read_options = twisterx::io::config::CSVReadOptions().UseThreads(false).BlockSize(1 << 30);
  if (!(status = Table::FromCSV("/tmp/csv1.csv", table1, read_options)).is_ok()) {
    LOG(ERROR) << "File read failed!";
    return 1;
  }
  if (!(status = Table::FromCSV("/tmp/csv2.csv", table2, read_options)).is_ok()) {
    LOG(ERROR) << "File read failed!";
    return 1;
  }
  LOG(INFO) << "Done reading tables";

  LOG(INFO) << "right join start";
  auto right_jc = JoinConfig::RightJoin(0, 0, JoinAlgorithm::HASH);
  RunJoin(right_jc, table1, table2, joined, "/tmp/h_out_right.csv");
  auto right_jc2 = JoinConfig::RightJoin(0, 0, JoinAlgorithm::SORT);
  RunJoin(right_jc2, table1, table2, joined, "/tmp/s_out_right.csv");
  LOG(INFO) << "right join end ----------------------------------";

  LOG(INFO) << "left join start";
  auto left_jc = JoinConfig::LeftJoin(0, 0, JoinAlgorithm::HASH);
  RunJoin(left_jc, table1, table2, joined, "/tmp/h_out_left.csv");
  auto left_jc2 = JoinConfig::LeftJoin(0, 0, JoinAlgorithm::SORT);
  RunJoin(left_jc2, table1, table2, joined, "/tmp/s_out_left.csv");
  LOG(INFO) << "left join end ----------------------------------";

  LOG(INFO) << "inner join start";
  auto inner_jc = JoinConfig::InnerJoin(0, 0, JoinAlgorithm::HASH);
  RunJoin(inner_jc, table1, table2, joined, "/tmp/h_out_inner.csv");
  auto inner_jc2 = JoinConfig::InnerJoin(0, 0, JoinAlgorithm::SORT);
  RunJoin(inner_jc2, table1, table2, joined, "/tmp/s_out_inner.csv");
  LOG(INFO) << "inner join end ----------------------------------";

  LOG(INFO) << "outer join start";
  auto outer_jc = JoinConfig::FullOuterJoin(0, 0, JoinAlgorithm::HASH);
  RunJoin(outer_jc, table1, table2, joined, "/tmp/h_out_outer.csv");
  auto outer_jc2 = JoinConfig::FullOuterJoin(0, 0, JoinAlgorithm::SORT);
  RunJoin(outer_jc2, table1, table2, joined, "/tmp/s_out_outer.csv");
  LOG(INFO) << "outer join end ----------------------------------";

  return 0;
}


