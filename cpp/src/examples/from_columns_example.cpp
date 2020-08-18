//
// Created by nira on 8/17/20.
//

#include <vector>
#include <iostream>

#include "column.hpp"

#define DATA_TYPE double

int main(int argc, char *argv[]) {
  const int size = 100;
  const int col_count = 2;
  std::vector<DATA_TYPE> cols[col_count];

  for (auto &col : cols) {
    col.resize(size);
    for (int i = 0; i < size; ++i) {
      col.push_back((DATA_TYPE) i);
    }
  }

  const std::shared_ptr<cylon::DataType>
      &dt = std::make_shared<cylon::DataType>(cylon::Type::DOUBLE);
  cylon::VectorColumn<double> cylon_col = cylon::VectorColumn<DATA_TYPE>("col0", dt, cols[0]);

  std::cout << "done " << cols[0].back() << cylon_col.GetID() << std::endl;

  return 0;
}