#ifndef TWISTERX_SRC_TWISTERX_JOIN_JOIN_CONFIG_H_
#define TWISTERX_SRC_TWISTERX_JOIN_JOIN_CONFIG_H_

namespace twisterx {
namespace join {
namespace config {

enum JoinType {
  INNER, LEFT, RIGHT
};
enum JoinAlgorithm {
  SORT, HASH
};

class JoinConfig {
 private:
  JoinType type = INNER;
  JoinAlgorithm algorithm = SORT;
  int left_column_idx, right_column_idx;

 public:
  JoinConfig(JoinType type, int left_column_idx, int right_column_idx) {
    this->type = type;
    this->left_column_idx = left_column_idx;
    this->right_column_idx = right_column_idx;
  }

  JoinConfig(JoinType type, int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
    this->type = type;
    this->left_column_idx = left_column_idx;
    this->right_column_idx = right_column_idx;
    this->algorithm = algorithm;
  }

 public:
  JoinConfig() = delete;

  static JoinConfig InnerJoin(int left_column_idx, int right_column_idx) {
    return JoinConfig(INNER, left_column_idx, right_column_idx);
  }

  static JoinConfig LeftJoin(int left_column_idx, int right_column_idx) {
    return JoinConfig(LEFT, left_column_idx, right_column_idx);
  }

  static JoinConfig RightJoin(int left_column_idx, int right_column_idx) {
    return JoinConfig(RIGHT, left_column_idx, right_column_idx);
  }

  static JoinConfig InnerJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
    return JoinConfig(INNER, left_column_idx, right_column_idx, algorithm);
  }

  static JoinConfig LeftJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
    return JoinConfig(LEFT, left_column_idx, right_column_idx, algorithm);
  }

  static JoinConfig RightJoin(int left_column_idx, int right_column_idx, JoinAlgorithm algorithm) {
    return JoinConfig(RIGHT, left_column_idx, right_column_idx, algorithm);
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
