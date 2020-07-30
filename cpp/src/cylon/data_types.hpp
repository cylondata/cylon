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

#ifndef CYLON_SRC_IO_DATATYPES_H_
#define CYLON_SRC_IO_DATATYPES_H_

namespace cylon {

/**
 * The types are a strip down from arrow types
 */
struct Type {
  enum type {
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
	/// Custom data type, implemented by user
	EXTENSION,
	/// Fixed size list of some logical type
	FIXED_SIZE_LIST,
	/// or nanoseconds.
	DURATION,
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
  };

  /**
   * Get the data layout
   * @return
   */
  Layout::layout getLayout() {
	return l;
  };
 private:
  // the type
  Type::type t;
  // the layout
  Layout::layout l;
};

}  // namespace cylon

#endif //CYLON_SRC_IO_DATATYPES_H_
