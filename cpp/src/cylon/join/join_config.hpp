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

#ifndef CYLON_SRC_CYLON_JOIN_JOIN_CONFIG_HPP_
#define CYLON_SRC_CYLON_JOIN_JOIN_CONFIG_HPP_

#include <utility>

#include "string"

namespace cylon {
namespace join {
namespace config {

enum JoinType {
  INNER, LEFT, RIGHT, FULL_OUTER
};
enum JoinAlgorithm {
  SORT, HASH
};

class JoinConfig {
 public:
  JoinConfig() = delete;

  JoinConfig(JoinType type, int left_column_idx, int right_column_idx)
      : JoinConfig(type, left_column_idx, right_column_idx, SORT) {
  }

  JoinConfig(JoinType type, int left_column_idx, int right_column_idx, JoinAlgorithm algorithm)
      : type(type), algorithm(algorithm), left_column_idx({left_column_idx}), right_column_idx({right_column_idx}) {}

  JoinConfig(JoinType type,
             int left_column_idx,
             int right_column_idx,
             JoinAlgorithm algorithm,
             std::string left_table_suffix,
             std::string right_table_suffix)
      : JoinConfig(type,
                   std::vector<int>{left_column_idx},
                   std::vector<int>{right_column_idx},
                   algorithm,
                   std::move(left_table_suffix),
                   std::move(right_table_suffix)) {}

  JoinConfig(JoinType type,
             std::vector<int> left_column_idx,
             std::vector<int> right_column_idx,
             JoinAlgorithm algorithm,
             std::string left_table_suffix,
             std::string right_table_suffix)
      : type(type),
        algorithm(algorithm),
        left_column_idx(std::move(left_column_idx)),
        right_column_idx(std::move(right_column_idx)),
        left_table_suffix(std::move(left_table_suffix)),
        right_table_suffix(std::move(right_table_suffix)) {
    if (left_column_idx.size() != right_column_idx.size()) {
      throw "left and right column indices sizes are not equal";
    }
  }

  static JoinConfig InnerJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm = SORT,
                              std::string left_table_suffix = "", std::string right_table_suffix = "") {
    return JoinConfig(INNER,
                      left_column_idx,
                      right_column_idx,
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }
  static JoinConfig InnerJoin(std::vector<int> left_column_idx,
                              std::vector<int> right_column_idx,
                              JoinAlgorithm algorithm = SORT,
                              std::string left_table_suffix = "",
                              std::string right_table_suffix = "") {
    return JoinConfig(INNER,
                      std::move(left_column_idx),
                      std::move(right_column_idx),
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }

  static JoinConfig LeftJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm = SORT,
                             std::string left_table_suffix = "", std::string right_table_suffix = "") {
    return JoinConfig(LEFT,
                      left_column_idx,
                      right_column_idx,
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }
  static JoinConfig LeftJoin(std::vector<int> left_column_idx,
                             std::vector<int> right_column_idx,
                             JoinAlgorithm algorithm = SORT,
                             std::string left_table_suffix = "",
                             std::string right_table_suffix = "") {
    return JoinConfig(LEFT,
                      std::move(left_column_idx),
                      std::move(right_column_idx),
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }

  static JoinConfig RightJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm = SORT,
                              std::string left_table_suffix = "", std::string right_table_suffix = "") {
    return JoinConfig(RIGHT,
                      left_column_idx,
                      right_column_idx,
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }
  static JoinConfig RightJoin(std::vector<int> left_column_idx,
                              std::vector<int> right_column_idx,
                              JoinAlgorithm algorithm = SORT,
                              std::string left_table_suffix = "",
                              std::string right_table_suffix = "") {
    return JoinConfig(RIGHT,
                      std::move(left_column_idx),
                      std::move(right_column_idx),
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }

  static JoinConfig FullOuterJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm = SORT,
                                  std::string left_table_suffix = "", std::string right_table_suffix = "") {
    return JoinConfig(FULL_OUTER,
                      left_column_idx,
                      right_column_idx,
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }
  static JoinConfig FullOuterJoin(std::vector<int> left_column_idx,
                                  std::vector<int> right_column_idx,
                                  JoinAlgorithm algorithm = SORT,
                                  std::string left_table_suffix = "",
                                  std::string right_table_suffix = "") {
    return JoinConfig(FULL_OUTER,
                      std::move(left_column_idx),
                      std::move(right_column_idx),
                      algorithm,
                      std::move(left_table_suffix),
                      std::move(right_table_suffix));
  }

  JoinType GetType() const {
    return type;
  }
  JoinAlgorithm GetAlgorithm() const {
    return algorithm;
  }

  const std::vector<int> &GetLeftColumnIdx() const {
    return left_column_idx;
  }
  const std::vector<int> &GetRightColumnIdx() const {
    return right_column_idx;
  }

  const std::string &GetLeftTableSuffix() const {
    return left_table_suffix;
  }
  const std::string &GetRightTableSuffix() const {
    return right_table_suffix;
  }

 private:
  JoinType type;
  JoinAlgorithm algorithm;
  const std::vector<int> left_column_idx, right_column_idx;
  const std::string left_table_suffix;
  const std::string right_table_suffix;
};
}  // namespace util
}  // namespace join
}  // namespace cylon

#endif //CYLON_SRC_CYLON_JOIN_JOIN_CONFIG_HPP_
