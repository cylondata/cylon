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

#ifndef GCYLON_ALL2ALL_UTIL_H
#define GCYLON_ALL2ALL_UTIL_H

#include <glog/logging.h>
#include <cuda.h>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

namespace gcylon {

/**
 * get one scalar value from device to host
 * @param buff
 * @return
 */
template <typename T>
inline T getScalar(const T * buff) {
  T value;
  cudaMemcpy(&value, buff, sizeof(T), cudaMemcpyDeviceToHost);
  return value;
}

/**
 * get part of a constant-size-type column from gpu to cpu
 * @tparam T
 * @param cv
 * @param start
 * @param end
 * @return
 */
template <typename T>
T * getColumnPart(const cudf::column_view &cv, int64_t start, int64_t end) {
  int64_t size = end - start;
  // data type size
  int dts = sizeof(T);
  uint8_t * host_array = new uint8_t[size * dts];
  cudaMemcpy(host_array, cv.data<uint8_t>() + start * dts, size * dts, cudaMemcpyDeviceToHost);
  return (T *) host_array;
}

/**
 * get top N elements of a constant-size-type column
 * @tparam T
 * @param cv
 * @param topN
 * @return
 */
template <typename T>
T * getColumnTop(const cudf::column_view &cv, int64_t topN = 5) {
  return getColumnPart<T>(cv, 0, topN);
}

/**
 * get tail N elements of a constant-size-type column
 * @tparam T
 * @param cv
 * @param tailN
 * @return
 */
template <typename T>
T * getColumnTail(const cudf::column_view &cv, int64_t tailN = 5) {
  return getColumnPart<T>(cv, cv.size() - tailN, cv.size());
}

/**
 * whether two cudf tables are equal with all elements in them
 * first sort both tables and compare then afterward
 * @param tv1
 * @param tv2
 * @return
 */
bool table_equal_with_sorting(cudf::table_view & tv1, cudf::table_view & tv2);

/**
 * whether two cudf tables are equal with all elements in them
 * @param tv1
 * @param tv2
 * @return
 */
bool table_equal(const cudf::table_view & tv1, const cudf::table_view & tv2);

/**
 * convert a vector of elements to a string with comma + space in between
 * @tparam T
 * @param vec
 * @return
 */
template<typename T>
std::string vectorToString(const std::vector<T> &vec) {
  if (vec.empty()) {
    return std::string();
  }

  std::ostringstream oss;
  // Convert all but the last element to avoid a trailing ","
  std::copy(vec.begin(), vec.end()-1,
            std::ostream_iterator<T>(oss, ", "));

  // Now add the last element with no delimiter
  oss << vec.back();
  return oss.str();
}

/**
 * convert a vector of cudf tables to table_views
 * @param tables
 * @return
 */
std::vector<cudf::table_view> tablesToViews(const std::vector<std::unique_ptr<cudf::table>> &tables);

} // end of namespace gcylon

#endif //GCYLON_ALL2ALL_UTIL_H
