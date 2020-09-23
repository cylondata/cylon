//
// Created by niranda on 9/22/20.
//
#include <groupby/groupby_hash.hpp>
#include <iostream>
#include <memory>
#include <arrow/api.h>

using namespace std;
int main(int argc, char *argv[]) {

  int a = 69;
  int b = 54;

//  std::tuple<int> res = {0};
//  cylon::UnaryFuncPtr<int, int> fn = cylon::SumUnaryFunc<int>;
//  fn(a, &res);
//  fn(b, &res);
//  cout << "b " << std::get<0>(res) << endl;

  using KERNEL = typename cylon::AggregateKernel<int, cylon::GroupByAggregationOp::MEAN>;
  KERNEL::HashMapType res = KERNEL::Init(a);

//  KERNEL::Update(a, &res);
  KERNEL::Update(b, &res);
  cout << "b " << KERNEL::Finalize(&res) << " " << sizeof(arrow::DoubleBuilder) << endl;

//  auto state = std::shared_ptr<cylon::State<long, long>>(new cylon::CountState<long>());
//  state->Update(&a);
//  state->Update(&b);
//
//  cout << "b " << state->GetResult()
//      << " " << sizeof(cylon::State<double>)
//      << " " << sizeof(cylon::SumState<double>)
//      << " " << sizeof(std::tuple<double, double>)
//      << " " << sizeof(std::tuple<double>)
//      << " " << sizeof(cylon::temp<int, int>)
//      << " " << sizeof(cylon::temp1<int, int>)
//      << endl;

  return 0;
}