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

using namespace twisterx;
using namespace twisterx::join::config;

int main(int argc, char *argv[]) {

  std::shared_ptr<Table> table1, table2, joined, expected_result;
  Status status;

  status = Table::FromCSV("/tmp/csv.csv", &table1,
						  io::config::CSVReadOptions().WithDelimiter(','));
  status = Table::FromCSV("/tmp/csv.csv", &table2);

  std::cout << "### right join start" << std::endl;
  auto jc1 = JoinConfig::RightJoin(0, 1, JoinAlgorithm::SORT);
  status = table1->Join(table2, jc1, &expected_result);
  expected_result->print();

  std::cout << "---" << std::endl;

  auto jc2 = JoinConfig::RightJoin(0, 1, JoinAlgorithm::HASH);
  status = table1->Join(table2, jc2, &joined);
  joined->print();

  std::cout << "### left join start" << std::endl;
  auto jc3 = JoinConfig::LeftJoin(0, 1, JoinAlgorithm::SORT);
  status = table1->Join(table2, jc3, &expected_result);
  expected_result->print();

  std::cout << "---" << std::endl;

  auto jc4 = JoinConfig::LeftJoin(0, 1, JoinAlgorithm::HASH);
  status = table1->Join(table2, jc4, &joined);
  joined->print();

  std::cout << "### inner join start" << std::endl;
  auto jc5 = JoinConfig::InnerJoin(0, 1, JoinAlgorithm::SORT);
  status = table1->Join(table2, jc5, &expected_result);
  expected_result->print();

  std::cout << "---" << std::endl;

  auto jc6 = JoinConfig::InnerJoin(0, 1, JoinAlgorithm::HASH);
  status = table1->Join(table2, jc6, &joined);
  joined->print();

  std::cout << "### outer join start" << std::endl;
  auto jc7 = JoinConfig::FullOuterJoin(0, 1, JoinAlgorithm::SORT);
  status = table1->Join(table2, jc7, &expected_result);
  expected_result->print();

  std::cout << "---" << std::endl;

  auto jc8 = JoinConfig::FullOuterJoin(0, 1, JoinAlgorithm::HASH);
  status = table1->Join(table2, jc8, &joined);
  joined->print();

//  joined->WriteCSV("/tmp/out.csv");

  return 0;
}


