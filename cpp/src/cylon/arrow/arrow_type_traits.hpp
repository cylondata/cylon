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

#ifndef CYLON_CPP_SRC_CYLON_ARROW_ARROW_TYPE_TRAITS_HPP_
#define CYLON_CPP_SRC_CYLON_ARROW_ARROW_TYPE_TRAITS_HPP_

namespace cylon {

template<typename ArrowT, typename Enable = void>
struct ArrowTypeTraits {};

template<typename ArrowT>
struct ArrowTypeTraits<ArrowT, arrow::enable_if_has_c_type<ArrowT>> {
  using ScalarT = typename arrow::TypeTraits<ArrowT>::ScalarType;
  using ArrayT = typename arrow::TypeTraits<ArrowT>::ArrayType;
  using ValueT = typename ArrowT::c_type;

  static ValueT ExtractFromScalar(const std::shared_ptr<arrow::Scalar> &scalar) {
    return std::static_pointer_cast<ScalarT>(scalar)->value;
  }
};

template<typename ArrowT>
struct ArrowTypeTraits<ArrowT, arrow::enable_if_has_string_view<ArrowT>> {
  using ScalarT = typename arrow::TypeTraits<ArrowT>::ScalarType;
  using ArrayT = typename arrow::TypeTraits<ArrowT>::ArrayType;
  using ValueT = arrow::util::string_view;

  static ValueT ExtractFromScalar(const std::shared_ptr<arrow::Scalar> &scalar) {
    return ValueT(*(std::static_pointer_cast<ScalarT>(scalar))->value);
  }
};
} // cylon

#endif //CYLON_CPP_SRC_CYLON_ARROW_ARROW_TYPE_TRAITS_HPP_
