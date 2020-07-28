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
 private:
  JoinType type;
  JoinAlgorithm algorithm;
  int left_column_idx, right_column_idx;

 public:
  JoinConfig() = delete;

  JoinConfig(JoinType type, int left_column_idx, int right_column_idx)
	  : JoinConfig(type, left_column_idx, right_column_idx, SORT) {
  }

  JoinConfig(JoinType type, int left_column_idx, int right_column_idx, JoinAlgorithm algorithm)
	  : type(type), algorithm(algorithm), left_column_idx(left_column_idx), right_column_idx(right_column_idx) {}

  static JoinConfig InnerJoin(int left_column_idx, int right_column_idx) {
	return {INNER, left_column_idx, right_column_idx};
  }

  static JoinConfig LeftJoin(int left_column_idx, int right_column_idx) {
	return {LEFT, left_column_idx, right_column_idx};
  }

  static JoinConfig RightJoin(int left_column_idx, int right_column_idx) {
	return {RIGHT, left_column_idx, right_column_idx};
  }

  static JoinConfig FullOuterJoin(int left_column_idx, int right_column_idx) {
	return {FULL_OUTER, left_column_idx, right_column_idx};
  }

  static JoinConfig InnerJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
	return {INNER, left_column_idx, right_column_idx, algorithm};
  }

  static JoinConfig LeftJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
	return {LEFT, left_column_idx, right_column_idx, algorithm};
  }

  static JoinConfig RightJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
	return {RIGHT, left_column_idx, right_column_idx, algorithm};
  }

  static JoinConfig FullOuterJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
	return {FULL_OUTER, left_column_idx, right_column_idx, algorithm};
  }

  JoinType GetType() const {
	return type;
  }
  JoinAlgorithm GetAlgorithm() const {
	return algorithm;
  }
  int GetLeftColumnIdx() const {
	return left_column_idx;
  }
  int GetRightColumnIdx() const {
	return right_column_idx;
  }
};
}  // namespace util
}  // namespace join
}  // namespace cylon

#endif //CYLON_SRC_CYLON_JOIN_JOIN_CONFIG_HPP_
