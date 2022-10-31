//
// Created by Mills "Bud" Staylor on 10/17/22.
//

#ifndef CYLON_SRC_CYLON_WINDOW_WINDOW_CONFIG_HPP_
#define CYLON_SRC_CYLON_WINDOW_WINDOW_CONFIG_HPP_

#include <utility>
#include <string>

namespace cylon {
namespace windowing {
namespace config {

enum LabelPosition {
  LEFT,
  CENTER,
  RIGHT
};

enum WindowOperation {
  ROLLING,
  WEIGHTED,
  EXPANDING,
  EXP_WEIGHTED
};

class WindowConfig {
 public:
  WindowConfig() = delete;

  WindowConfig(const WindowOperation window_operation, const int observations) : window_operation(window_operation),
                                                                     observations(observations) {

  }

  WindowConfig(const WindowOperation window_operation, const std::string offset) : window_operation(window_operation),
                                                                       offset(std::move(offset)) {}
  WindowOperation GetWindowOperation() const {
    return window_operation;
  }
  int GetObservations() const {
    return observations;
  }
  const std::string &GetOffset() const {
    return offset;
  }
  LabelPosition GetLabelPosition() const {
    return label_position;
  }
  int GetMinPeriods() const {
    return min_periods;
  }
  const int GetStep() const {
    return step;
  }

  /**
   * Creates a Windowing Configuration applying defaults and is based on the number of
   * observations per window
   * @param observations
   * @return
   */
  static WindowConfig DefaultWithObservations(const WindowOperation window_operation, const int observations) {
    return WindowConfig(window_operation, observations);
  }

  /**
   * Creates a Windowing Configuration applying defaults and is based on an offset or
   * a time period for each window.
   * @param observations
   * @return
   */
  static WindowConfig DefaultWithOffset(const WindowOperation window_operation, const std::string offset) {
    return WindowConfig(window_operation, offset);
  }

 private:
  const WindowOperation window_operation;
  const int observations{};
  const std::string offset{};
  const LabelPosition label_position = LabelPosition::RIGHT;
  const int min_periods{};
  const int step{};
};
}
}
}

#endif //CYLON_SRC_CYLON_WINDOW_WINDOW_CONFIG_HPP_
