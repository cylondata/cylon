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


#ifndef CYLON_CPP_SRC_CYLON_NET_MPI_MPI_TYPE_TRAITS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_MPI_MPI_TYPE_TRAITS_HPP_

#include <mpi.h>

namespace cylon {

template<typename T>
struct MPICTypeTraits {};

template<>
struct MPICTypeTraits<uint8_t> {
  static MPI_Datatype MPIDataType() { return MPI_UINT8_T; }
};

template<>
struct MPICTypeTraits<int8_t> {
  static MPI_Datatype MPIDataType() { return MPI_INT8_T; }
};

template<>
struct MPICTypeTraits<uint16_t> {
  static MPI_Datatype MPIDataType() { return MPI_UINT16_T; }
};

template<>
struct MPICTypeTraits<int16_t> {
  static MPI_Datatype MPIDataType() { return MPI_INT16_T; }
};

template<>
struct MPICTypeTraits<uint32_t> {
  static MPI_Datatype MPIDataType() { return MPI_UINT32_T; }
};

template<>
struct MPICTypeTraits<int32_t> {
  static MPI_Datatype MPIDataType() { return MPI_INT32_T; }
};

template<>
struct MPICTypeTraits<uint64_t> {
  static MPI_Datatype MPIDataType() { return MPI_UINT64_T; }
};

template<>
struct MPICTypeTraits<int64_t> {
  static MPI_Datatype MPIDataType() { return MPI_INT64_T; }
};

template<>
struct MPICTypeTraits<float> {
  static MPI_Datatype MPIDataType() { return MPI_FLOAT; }
};

template<>
struct MPICTypeTraits<double> {
  static MPI_Datatype MPIDataType() { return MPI_DOUBLE; }
};

}

#endif //CYLON_CPP_SRC_CYLON_NET_MPI_MPI_TYPE_TRAITS_HPP_
