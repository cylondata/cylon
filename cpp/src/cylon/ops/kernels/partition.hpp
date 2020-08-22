#ifndef CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_

#include <status.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <table.hpp>
namespace cylon {
namespace kernel {
Status HashPartition(CylonContext *ctx, const std::shared_ptr<Table> table,
                     const std::vector<int> &hash_columns, int no_of_partitions,
                     std::unordered_map<int, std::shared_ptr<Table>> *out);
}
}

#endif //CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_
