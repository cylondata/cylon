#ifndef CYLON_SRC_CYLON_OPS_EXECUTION_HPP_
#define CYLON_SRC_CYLON_OPS_EXECUTION_HPP_
namespace cylon {
class Execution {
  virtual bool IsComplete() = 0;
  void WaitForCompletion() {
    while (!this->IsComplete()) {

    }
  }
};
}
#endif //CYLON_SRC_CYLON_OPS_EXECUTION_HPP_
