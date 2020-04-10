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
  JoinConfig(int left_column_idx, int right_column_idx) {
    this->left_column_idx = left_column_idx;
    this->right_column_idx = right_column_idx;
  }

 public:
  JoinConfig() = delete;

  static JoinConfig *InnerJoin(int left_column_idx, int right_column_idx) {
    return new JoinConfig(left_column_idx, right_column_idx);
  }

  static JoinConfig *LeftJoin(int left_column_idx, int right_column_idx) {
    return new JoinConfig(left_column_idx, right_column_idx);
  }

  static JoinConfig *RightJoin(int left_column_idx, int right_column_idx) {
    return new JoinConfig(left_column_idx, right_column_idx);
  }

  JoinConfig *UseHashAlgorithm() {
    this->algorithm = HASH;
    return this;
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
