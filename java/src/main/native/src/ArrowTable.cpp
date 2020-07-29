#include <org_cylondata_cylon_arrow_ArrowTable.h>
#include <cstdint>
#include <iostream>
#include "arrow/arrow_builder.hpp"
#include "Utils.hpp"

enum JavaType {
  NONE = 0,
  Null = 1,
  Int = 2,
  FloatingPoint = 3,
  Binary = 4,
  Utf8 = 5,
  Bool = 6,
  Decimal = 7,
  Date = 8,
  Time = 9,
  Timestamp = 10,
  Interval = 11,
  List = 12,
  Struct_ = 13,
  Union = 14,
  FixedSizeBinary = 15,
  FixedSizeList = 16,
  Map = 17,
  Duration = 18,
  LargeBinary = 19,
  LargeUtf8 = 20,
  LargeList = 21,
};
enum CppType {
  /// A NULL type having no physical storage
  NA,

  /// Boolean as 1 bit, LSB bit-packed ordering
  BOOL,

  /// Unsigned 8-bit little-endian integer
  UINT8,

  /// Signed 8-bit little-endian integer
  INT8,

  /// Unsigned 16-bit little-endian integer
  UINT16,

  /// Signed 16-bit little-endian integer
  INT16,

  /// Unsigned 32-bit little-endian integer
  UINT32,

  /// Signed 32-bit little-endian integer
  INT32,

  /// Unsigned 64-bit little-endian integer
  UINT64,

  /// Signed 64-bit little-endian integer
  INT64,

  /// 2-byte floating point value
  HALF_FLOAT,

  /// 4-byte floating point value
  FLOAT,

  /// 8-byte floating point value
  DOUBLE,

  /// UTF8 variable-length string as List<Char>
  STRING,

  /// Variable-length bytes (no guarantee of UTF8-ness)
  BINARY,

  /// Fixed-size binary. Each value occupies the same number of bytes
  FIXED_SIZE_BINARY,

  /// int32_t days since the UNIX epoch
  DATE32,

  /// int64_t milliseconds since the UNIX epoch
  DATE64,

  /// Exact timestamp encoded with int64 since UNIX epoch
  /// Default unit millisecond
  TIMESTAMP,

  /// Time as signed 32-bit integer, representing either seconds or
  /// milliseconds since midnight
  TIME32,

  /// Time as signed 64-bit integer, representing either microseconds or
  /// nanoseconds since midnight
  TIME64,

  /// YEAR_MONTH or DAY_TIME interval in SQL style
  INTERVAL,

  /// Precision- and scale-based decimal type. Storage type depends on the
  /// parameters.
  DECIMAL,

  /// A list of some logical data type
  LIST,

  /// Struct of logical types
  STRUCT,

  /// Unions of logical types
  UNION,

  /// Dictionary-encoded type, also called "categorical" or "factor"
  /// in other programming languages. Holds the dictionary value
  /// type but not the dictionary itself, which is part of the
  /// ArrayData struct
  DICTIONARY,

  /// Map, a repeated struct logical type
  MAP,

  /// Custom data type, implemented by user
  EXTENSION,

  /// Fixed size list of some logical type
  FIXED_SIZE_LIST,

  /// Measure of elapsed time in either seconds, milliseconds, microseconds
  /// or nanoseconds.
  DURATION,

  /// Like STRING, but with 64-bit offsets
  LARGE_STRING,

  /// Like BINARY, but with 64-bit offsets
  LARGE_BINARY,

  /// Like LIST, but with 64-bit offsets
  LARGE_LIST
};

CppType ToArrowType(JNIEnv *env, JavaType java_type, const std::string &col_name) {
  switch (java_type) {
    case NONE:return CppType::NA;
    case Null:return CppType::NA;
    case Int:return CppType::INT32;
    case FloatingPoint: return CppType::FLOAT;
    case Binary:return CppType::BINARY;
    case Utf8:return CppType::STRING;
    case Bool:return CppType::BOOL;
    case Decimal:return CppType::DECIMAL;
    case Date:return CppType::DATE64;
    case Time:return CppType::TIME64;
    case Timestamp:return CppType::TIMESTAMP;
    case Interval:return CppType::INTERVAL;
    case List:return CppType::LIST;
    case Struct_:return CppType::STRUCT;
    case Union:return CppType::UNION;
    case FixedSizeBinary:return CppType::FIXED_SIZE_BINARY;
    case FixedSizeList:return CppType::FIXED_SIZE_LIST;
    case Map:return CppType::MAP;
    case Duration:return CppType::DURATION;
    case LargeBinary:return CppType::LARGE_BINARY;
    case LargeUtf8:return CppType::LARGE_STRING;
    case LargeList:return CppType::LARGE_LIST;
    default:throwCylonRuntimeException(env, "Unsupported data type for column " + col_name);
  }
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_arrow_ArrowTable_addColumn
    (JNIEnv *env, jclass clz, jstring table_id, jint data_type, jlong address, jlong size) {
  auto *buffer = reinterpret_cast<int8_t *>(address);
  for (int64_t i = 0; i < size;) {
    int a = int((unsigned char) (buffer[i + 3]) << 24 |
        (unsigned char) (buffer[i + 2]) << 16 |
        (unsigned char) (buffer[i + 1]) << 8 |
        (unsigned char) (buffer[i]));
    std::cout << a << ",";
    i += 4;
  }
  std::cout << std::endl;
}

void checkStatusAndThrow(JNIEnv *env, cylon::Status status) {
  if (!status.is_ok()) {
    throwCylonRuntimeException(env, status.get_msg());
  }
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_arrow_ArrowTable_createTable
    (JNIEnv *env, jclass cls, jstring table_id) {
  checkStatusAndThrow(env, cylon::cyarrow::BeginTable(jstr_to_str(env, table_id)));
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_arrow_ArrowTable_addColumn
    (JNIEnv *env, jclass cls, jstring table_id, jstring col_name, jbyte type_id,
     jint value_count,
     jint null_count,
     jlong validity_address, jlong validity_size,
     jlong data_address, jlong data_size) {
  auto c_column_name = jstr_to_str(env, col_name);
  auto arrow_type = ToArrowType(env, static_cast<JavaType>(type_id), c_column_name);
  checkStatusAndThrow(env, cylon::cyarrow::AddColumn(
      jstr_to_str(env, table_id),
      c_column_name,
      arrow_type,
      value_count,
      null_count,
      validity_address,
      validity_size,
      data_address,
      data_size
  ));
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_arrow_ArrowTable_finishTable
    (JNIEnv *env, jclass cls, jstring table_id) {
  checkStatusAndThrow(env, cylon::cyarrow::FinishTable(jstr_to_str(env, table_id)));
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_arrow_ArrowTable_createTable
    (JNIEnv *env, jclass clz, jstring tab_id, jbyteArray schema_bytes) {
//  std::cout << "Creating table..." << std::endl;
//  auto schema_length = env->GetArrayLength(schema_bytes);
//  auto *schema_buffer = new uint8_t[schema_length];
//  std::cout << "Getting byte array" << std::endl;
//  env->GetByteArrayRegion(schema_bytes, 0, schema_length, reinterpret_cast<jbyte *>(schema_bytes));
//
//  std::cout << "Calling build" << std::endl;
//  cylon::cyarrow::Build("", schema_buffer, schema_length,
//                       std::vector<int8_t *>{}, std::vector<int64_t>{});
//  std::cout << "Ot of  build" << std::endl;
//  delete[] schema_buffer;
}