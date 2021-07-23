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

#endif //GCYLON_ALL2ALL_UTIL_H
