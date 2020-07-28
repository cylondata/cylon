#include "test_header.hpp"

TEST_CASE( "Join testing", "[join]" ) {
//  std::shared_ptr<Table> table1, table2, joined;
//
//  auto read_options = cylon::io::config::CSVReadOptions().UseThreads(false);
//  Table::FromCSV(ctx, {
//      "../data/join/table0.csv",
//      "../data/join/tabl1.csv"
//	}, {&table1, &table2}, read_options);
//
//  auto join_config = cylon::join::config::JoinConfig::InnerJoin(0, 0);
//  table1->DistributedJoin(table2, join_config, &joined);

  LOG(INFO)<<"^^^^^^^^^^^^^ TEST";
  LOG(INFO)<<"^^^^^^^^^^^^^ TEST";
  REQUIRE(true);

  // TODO read results from ../data/join/result.csv and compare with joined
}
