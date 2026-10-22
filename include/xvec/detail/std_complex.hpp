//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <complex>

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>

namespace _XVEC_NAMESPACE::simd
{

/**
 * @defgroup cpp26_complex_math C++26: Complex math
 * @brief Element-wise mathematical functions for complex vectors.
 *
 * @see https://eel.is/c++draft/simd.complex.math
 * @{
 */

/// @brief Compute the conjugate of each complex element.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A vector containing the conjugate of each input element.
template<vec_complex _Vc> constexpr _Vc conj(const _Vc& value) { return detail::conj(target, value); }

/// @brief Extract the real component of each complex element.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A real-valued vector containing the corresponding real components.
template<vec_complex _Vc> constexpr auto real(const _Vc& value) { return value.real(); }

/// @brief Extract the imaginary component of each complex element.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A real-valued vector containing the corresponding imaginary components.
template<vec_complex _Vc> constexpr auto imag(const _Vc& value) { return value.imag(); }

/// @brief Compute the squared magnitude of each complex element.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A real-valued vector containing the squared magnitude of each input element.
template<vec_complex _Vc> constexpr auto norm(const _Vc& value) { return (value * conj(value)).real(); }

/// @brief Compute the magnitude of each complex element.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A real-valued vector containing the magnitude of each input element.
template<vec_complex _Vc> constexpr auto abs(const _Vc& value) { return sqrt(norm(value)); }

/// @} cpp26_complex_math

/// @addtogroup future_math
/// @{

/// @brief Round the real and imaginary components of each element upward.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A complex vector with each component rounded toward positive infinity.
template<vec_complex _Vc> inline _Vc ceil(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(ceil(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }

/// @brief Round the real and imaginary components of each element downward.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A complex vector with each component rounded toward negative infinity.
template<vec_complex _Vc> inline _Vc floor(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(floor(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }

/// @brief Truncate the real and imaginary components of each element.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A complex vector with each component rounded toward zero.
template<vec_complex _Vc> inline _Vc trunc(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(trunc(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }

/// @brief Apply round independently to the real and imaginary components.
/// @tparam _Vc The complex vector type.
/// @param value The input vector.
/// @return A complex vector containing the rounded components.
template<vec_complex _Vc> inline _Vc round(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(round(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }

/// @} future_math

/// :TODO: Implement arg, proj, polar, exp, log, log10, [a]sin[h], [a]cos[h], [a]tan[h], pow, sqrt.

} // namespace _XVEC_NAMESPACE::simd

namespace _XVEC_NAMESPACE {
  // When implemented, arg, proj, polar, exp, etc.
  using simd::conj;
  using simd::imag;
  using simd::real;
  using simd::norm;
}
