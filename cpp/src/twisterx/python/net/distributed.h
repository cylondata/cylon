//
// Created by vibhatha on 4/25/20.
//

#ifndef TWISTERX_SRC_TWISTERX_PYTHON_NET_DISTRIBUTED_H_
#define TWISTERX_SRC_TWISTERX_PYTHON_NET_DISTRIBUTED_H_
#include <iostream>
#include "../../net/ops/all_to_all.hpp"
#include "../../arrow/arrow_all_to_all.hpp"

namespace twisterx {
namespace net {
void cdist_init();
int cget_rank();
int cget_size();
void call_to_all(void *buffer, int length);
void cdist_finalize();
}
}

#endif //TWISTERX_SRC_TWISTERX_PYTHON_NET_DISTRIBUTED_H_
