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

/// @brief Compute the conjugate of every value (i.e., negate the imaginary part).
/// \ingroup simd_complex
/// @tparam _Vc The vec of complex values.
/// @param value The vec value (complex or real) from which to compute the conjugate.
/// @return A vec in which each complex element is the conjugate of the respective input element.
template<vec_complex _Vc> constexpr _Vc conj(const _Vc& value) { return detail::conj(target, value); }

/// @brief Extract the real components of all the complex elements as a vec value.
/// \ingroup simd_complex
/// @tparam _Vc The vec of complex values.
/// @param value The vec value from which to extract the real elements.
/// @return A vec containing only the real components of every equivalent complex element from the input vec.
template<vec_complex _Vc> constexpr auto real(const _Vc& value) { return value.real(); }

/// @brief Extract the imaginary components of all the complex elements as a vec value.
/// \ingroup simd_complex
/// @tparam _Vc The vec of complex values.
/// @param value The vec value from which to extract the imaginary elements.
/// @return A vec containing only the imaginary components of every equivalent complex element from the input vec.
template<vec_complex _Vc> constexpr auto imag(const _Vc& value) { return value.imag(); }

/// @brief Compute the square of the magnitude of the complex value as a real valued vec.
/// \ingroup simd_complex
/// @tparam _Vc The vec of complex values.
/// @param value The value from which to compute the norm.
/// @return A real-valued vec where every element is the norm of the corresponding input complex vec value.
template<vec_complex _Vc> constexpr auto norm(const _Vc& value) { return (value * conj(value)).real(); }

/// @brief Compute the square of the magnitude of the complex value as a real valued vec.
/// \ingroup simd_complex
/// @tparam _Vc The vec of complex values.
/// @param value The value from which to compute the norm.
/// @return A real-valued vec where every element is the norm of the corresponding input complex vec value.
template<vec_complex _Vc> constexpr auto abs(const _Vc& value) { return sqrt(norm(value)); }

/// @brief Rounding operations implemented for complex numbers. Both the real and imaginary components will be rounded.
/// @{
/// @ingroup simd_complex
template<vec_complex _Vc> inline _Vc ceil(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(ceil(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }
template<vec_complex _Vc> inline _Vc floor(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(floor(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }
template<vec_complex _Vc> inline _Vc trunc(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(trunc(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }
template<vec_complex _Vc> inline _Vc round(const _Vc& value)
 { return simd_bit_cast<typename _Vc::value_type>(round(simd_bit_cast<typename _Vc::value_type::value_type>(value))); }
///@}

/// :TODO: Implement arg, proj, polar, exp, log, log10, [a]sin[h], [a]cos[h], [a]tan[h], pow, sqrt.

} // namespace _XVEC_NAMESPACE::simd

namespace _XVEC_NAMESPACE {
  // When implemented, arg, proj, polar, exp, etc.
  using simd::conj;
  using simd::imag;
  using simd::real;
  using simd::norm;
}
