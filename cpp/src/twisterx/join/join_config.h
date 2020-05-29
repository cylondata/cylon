#ifndef TWISTERX_SRC_TWISTERX_JOIN_JOIN_CONFIG_H_
#define TWISTERX_SRC_TWISTERX_JOIN_JOIN_CONFIG_H_

namespace twisterx {
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
}
}
}

#endif //TWISTERX_SRC_TWISTERX_JOIN_JOIN_CONFIG_H_
