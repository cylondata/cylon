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

/**
 * get one scalar value from device to host
 * @param buff
 * @return
 */
template <typename T>
inline T getScalar(const uint8_t * buff) {
    uint8_t *hostArray= new uint8_t[sizeof(T)];
    cudaMemcpy(hostArray, buff, sizeof(T), cudaMemcpyDeviceToHost);
    T * hdata = (T *) hostArray;
    return hdata[0];
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
    uint8_t * hostArray = new uint8_t[size * dts];
    cudaMemcpy(hostArray, cv.data<uint8_t>() + start * dts, size * dts, cudaMemcpyDeviceToHost);
    return (T *) hostArray;
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

#endif //GCYLON_ALL2ALL_UTIL_H
