#ifndef TWISTERX_SRC_IO_DATATYPES_H_
#define TWISTERX_SRC_IO_DATATYPES_H_

namespace twisterx {

struct Type {
  enum type {
    BIGINT = 0,
    BIT = 1,
    DATEDAY = 2,
    DATEMILLI = 3,
    DECIMAL = 4,
    DURATION = 5,
    EXTENSIONTYPE = 38,
    FIXED_SIZE_LIST = 6,
    FIXEDSIZEBINARY = 7,
    FLOAT4 = 8,
    FLOAT8 = 9,
    INT = 10,
    INTERVALDAY = 11,
    INTERVALYEAR = 12,
    LIST = 13,
    MAP = 14,
    NULL_ = 15,
    SMALLINT = 16,
    STRUCT = 17,
    TIMEMICRO = 18,
    TIMEMILLI = 19,
    TIMENANO = 20,
    TIMESEC = 21,
    TIMESTAMPMICRO = 22,
    TIMESTAMPMICROTZ = 23,
    TIMESTAMPMILLI = 24,
    TIMESTAMPMILLITZ = 25,
    TIMESTAMPNANO = 26,
    TIMESTAMPNANOTZ = 27,
    TIMESTAMPSEC = 28,
    TIMESTAMPSECTZ = 29,
    TINYINT = 30,
    UINT1 = 31,
    UINT2 = 32,
    UINT4 = 33,
    UINT8 = 34,
    UNION = 35,
    VARBINARY = 36,
    VARCHAR = 37
  };
};

/**
 * The layout of the data type
 */
struct Layout {
  enum layout {
    FIXED_WIDTH = 1,
    VARIABLE_WIDTH = 2
  };
};

/**
 * Base class for encapsulating a data type
 */
class DataType {
public:
  DataType() {}

  explicit DataType(Type::type t) : t(t), l(Layout::FIXED_WIDTH) {}

  DataType(Type::type t, Layout::layout l) : t(t), l(l) {}

  /**
   * The the type as an enum
   *
   * @return
   */
  Type::type getType() {
    return t;
  }

  /**
   * Get the data layout
   * @return
   */
  Layout::layout getLayout() {
    return l;
  }
private:
  // the type
  Type::type t;
  // the layout
  Layout::layout l;
};

/**
 * The numeric data types
 */
class NumericType : DataType {
public:
private:

};

/**
 * The list type
 */
class ListType : DataType {

};

}

#endif //TWISTERX_SRC_IO_DATATYPES_H_
