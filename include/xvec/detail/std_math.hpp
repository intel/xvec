//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <concepts>
#include <limits> // std::numeric_limits
#include <type_traits>

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>

#if defined(__INTEL_CLANG_COMPILER) && !defined(_XVEC_FORCE_SCALAR)
  #include <xvec/x86/math.hpp>
#else
  #include <xvec/generic/math.hpp>
#endif

namespace _XVEC_NAMESPACE::simd
{

/**
 * @defgroup cpp26_math C++26: Mathematical functions
 * @brief Element-wise mathematical functions for SIMD vectors.
 *
 * @see https://eel.is/c++draft/simd.math
 */

/**
 * @defgroup future_math Additional mathematical functions
 * @brief Additional approximate, fused, integral-classification, and complex operations.
 *
 * These are xvec extensions, not a claim of adoption into a future standard.
 */

/// @addtogroup cpp26_math
/// @{

/// @brief Compute the absolute value of each floating-point element.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the absolute values.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> abs(const _Vp& v) { return detail::abs(target, v); }

/// @brief Compute the absolute value of each signed-integral element.
/// @ingroup future_math
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the absolute values.
template<vec_signed_integral _Vp> constexpr _Vp abs(const _Vp& v) { return detail::abs(target, v); }

/// @brief Combine corresponding magnitudes and signs.
/// @tparam _Vp The input vector type.
/// @param magnitude The vector supplying the magnitudes.
/// @param sign The vector supplying the signs.
/// @return A vector with each magnitude from magnitude and each sign from sign.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> copysign(const _Vp& magnitude, const _Vp& sign)
{
  using detail::vec_as_container;

  // It must be some sort of a float. Combine the sign bit from one parameter with the magnitude bits from the
  // other parameter. This will turn into a single ternary logic instruction on machines that support Intel AVX-512.
  const auto tAbs = vec_as_container(abs(magnitude));
  constexpr auto signBit = vec_as_container(_Vp(typename _Vp::value_type(-0.0)));
  const auto tSign = vec_as_container(sign) & signBit;
  return std::bit_cast<_Vp>(tAbs | tSign);
}

/// @} cpp26_math

/// @addtogroup future_math
/// @{

/// @brief Compute the approximate reciprocal of each element.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing approximate reciprocals.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> rcp (const _Vp& v) { return detail::rcp(target, v); }

/// @brief Compute the approximate reciprocal square root of each element.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing approximate reciprocal square roots.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> rsqrt(const _Vp& v) { return detail::rsqrt(target, v); }

/// @} future_math

/// @addtogroup cpp26_math
/// @{

/// @brief Compute the square root of each element.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the square roots.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> sqrt (const _Vp& v) { return detail::sqrt(target, v); }

/// @brief Compute (lhs * rhs) + acc element-wise with a single rounding.
/// @tparam _Vp The input vector type.
/// @param lhs The left multiplication operand.
/// @param rhs The right multiplication operand.
/// @param acc The value added to the product.
/// @return A vector containing the fused multiply-add results.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp>
  fma (const _Vp& lhs, const _Vp& rhs, const _Vp& acc) { return detail::fma(target, lhs, rhs, acc); }

/// @} cpp26_math

/// @addtogroup future_math
/// @{

/// @brief Compute the complex multiply-add operation element-wise.
/// @tparam _Vp The complex vector type.
/// @param lhs The left multiplication operand.
/// @param rhs The right multiplication operand.
/// @param acc The accumulator.
/// @return A vector containing the complex multiply-add results.
template<vec_complex _Vp>
constexpr _Vp fma (const _Vp& lhs, const _Vp& rhs, const _Vp& acc)
  { return detail::fma(target, lhs, rhs, acc); }

/// @brief Compute lhs * rhs - acc at even indexes and lhs * rhs + acc at odd indexes.
/// Each expression is computed with a single rounding to the result type.
/// @tparam _Vp The input vector type.
/// @param lhs The left multiplication operand.
/// @param rhs The right multiplication operand.
/// @param acc The value alternately subtracted or added.
/// @return A vector containing the alternating fused results.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> fmaddsub(const _Vp& lhs, const _Vp& rhs, const _Vp& acc)
  { return detail::fmaddsub(target, lhs, rhs, acc); }

/// @brief Compute lhs * rhs + acc at even indexes and lhs * rhs - acc at odd indexes.
/// Each expression is computed with a single rounding to the result type.
/// @tparam _Vp The input vector type.
/// @param lhs The left multiplication operand.
/// @param rhs The right multiplication operand.
/// @param acc The value alternately added or subtracted.
/// @return A vector containing the alternating fused results.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> fmsubadd(const _Vp& lhs, const _Vp& rhs, const _Vp& acc)
  { return detail::fmsubadd(target, lhs, rhs, acc); }

/// @} future_math

/// @addtogroup cpp26_math
/// @{

/// @brief Round each element toward positive infinity.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the rounded values.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> ceil(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_ceil)
    return __builtin_elementwise_ceil(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::CEIL>(target, v);
  #endif
}

/// @brief Round each element toward negative infinity.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the rounded values.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> floor(const _Vp& v)
{
  #if __has_builtin(__builtin_elementwise_floor)
    return __builtin_elementwise_floor(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::FLOOR>(target, v);
  #endif
}

/// @brief Round each element toward zero.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the truncated values.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> trunc(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_trunc)
    return __builtin_elementwise_trunc(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::TRUNC>(target, v);
  #endif
}

/// @brief Round each element to an integer using the current rounding mode.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the rounded values.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> rint(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_rint)
    return __builtin_elementwise_rint(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::RINT>(target, v);
  #endif
}

/// @brief Round each element to the nearest integer value.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the rounded values.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> round(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_roundeven)
    return __builtin_elementwise_round(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::ROUND>(target, v);
  #endif
}

/// @} cpp26_math

namespace detail
{
/// Generic implementation of NaN classification.
/// @internal
template<math_floating_point _Vp>
constexpr deduced_mask_t<_Vp> isnan(generic_tag, const _Vp& v)
{
  // Exponent mask is the same as infinity (infinite has no sign and no
  // fractional bits, just a fully populated exponent).
  constexpr auto expMask = vec_as_container(std::numeric_limits<_Vp>::infinity());

  // Fractional mask is the same as the largest subnormal number. This can be
  // found by getting the smallest normal number and subtracting one from its integer
  // representation.
  using _Tp = typename _Vp::value_type;
  constexpr auto fm = std::bit_cast<container_for_type<_Tp>>(std::numeric_limits<_Tp>::min()) - 1;
  constexpr auto fractionMask = typename _Vp::traits::element_container_type(fm);

  // Check for NaN by using the actual bit-pattern. The exponent must be all
  // ones, and the fraction nonzero. Checking a NaN against itself is another
  // way but is fragile with respect to compiler optimisations.
  const auto asC = vec_as_container(v);
  return ((asC & expMask) == expMask) & ((asC & fractionMask) != uint8_t(0));
}

/// Generic implementation of infinity classification.
/// @internal
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isinf(generic_tag tag, const _Vp& v)
  { return !isnan(tag, v) && (abs(v) > std::numeric_limits<_Vp>::max()); }

/// Generic implementation of finite-value classification.
/// @internal
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isfinite(generic_tag tag, const _Vp& v)
  { return !(isinf(tag, v) || isnan(tag, v)); }

/// Generic implementation of normal-value classification.
/// @internal
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isnormal(generic_tag, const _Vp& v)
{
  return (abs(v) >= std::numeric_limits<_Vp>::min()) &&
         (abs(v) <= std::numeric_limits<_Vp>::max());
}

/// Generic implementation of sign-bit classification.
/// @internal
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> signbit(generic_tag, const _Vp& v)
{
  constexpr auto signBit = vec_as_container(_Vp(typename _Vp::value_type(-0.0)));
  auto m = (vec_as_container(v) & signBit) != uint8_t(0);
  return detail::deduced_mask_t<_Vp>(m);
}

}

// namespace detail

/// @addtogroup cpp26_math
/// @{

/// @brief Identify elements representing positive or negative infinity.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A mask whose elements are true for infinite input elements.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isinf(const _Vp& v) { return detail::isinf(target, v); }

/// @brief Identify elements representing NaN.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A mask whose elements are true for NaN input elements.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isnan(const _Vp& v) { return detail::isnan(target, v); }

/// @brief Identify finite elements.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A mask whose elements are true for inputs that are neither infinite nor NaN.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isfinite(const _Vp& v) { return detail::isfinite(target, v); }

/// @brief Identify normal floating-point elements.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A mask whose elements are true for normal, nonzero finite inputs.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isnormal(const _Vp& v) { return detail::isnormal(target, v); }

/// @brief Identify elements whose sign bit is set.
/// This includes negative zero and negative values of infinity or NaN.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A mask containing the sign-bit state of each input element.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> signbit(const _Vp& v) {  return detail::signbit(target, v); }

/// @} cpp26_math

/// @brief Identify negative integral elements.
/// @ingroup future_math
/// @tparam _Vp The integral vector type.
/// @param v The input vector.
/// @return A mask whose elements are true for negative inputs.
/// All output elements are false for unsigned input types.
template<vec_integral _Vp>
constexpr typename _Vp::mask_type signbit(const _Vp& v) {
  if constexpr (vec_unsigned_integral<_Vp>)
    return typename _Vp::mask_type(false); // Unsigned integer can never become signed
  else
    return v < _Vp(); // Signed integer 
}

/// @addtogroup cpp26_math
/// @{

/// @brief Compare corresponding elements using greater-than.
/// @param x The left operand.
/// @param y The right operand.
/// @return A mask containing the element-wise comparison results.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isgreater(const _Vp& x, const _Vp& y) { return x > y; }

/// @brief Compare corresponding elements using greater-than-or-equal.
/// @param x The left operand.
/// @param y The right operand.
/// @return A mask containing the element-wise comparison results.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isgreaterequal(const _Vp& x, const _Vp& y) { return x >= y; }

/// @brief Compare corresponding elements using less-than.
/// @param x The left operand.
/// @param y The right operand.
/// @return A mask containing the element-wise comparison results.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isless(const _Vp& x, const _Vp& y) { return x < y; }

/// @brief Compare corresponding elements using less-than-or-equal.
/// @param x The left operand.
/// @param y The right operand.
/// @return A mask containing the element-wise comparison results.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> islessequal(const _Vp& x, const _Vp& y) { return x <= y; }

/// @brief Identify corresponding elements that compare less or greater.
/// @param x The left operand.
/// @param y The right operand.
/// @return A mask that is false for equal or unordered element pairs.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> islessgreater(const _Vp& x, const _Vp& y) { return (x < y) | (x > y); }

/// @brief Identify unordered element pairs.
/// @param x The left operand.
/// @param y The right operand.
/// @return A mask that is true where either corresponding input is NaN.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isunordered(const _Vp& x, const _Vp& y) { return isnan(x) | isnan(y); }

// Basic
// :TODO: Move from elsewhere in some cases.

// Exponential
_XVEC_UNARY_MATHS_OP(exp);
_XVEC_UNARY_MATHS_OP(exp2);
_XVEC_UNARY_MATHS_OP(expm1);
_XVEC_UNARY_MATHS_OP(log);
_XVEC_UNARY_MATHS_OP(log2);
_XVEC_UNARY_MATHS_OP(log1p);

// Trig and hyperbolic
_XVEC_UNARY_MATHS_OP(sin); _XVEC_UNARY_MATHS_OP(asin);
_XVEC_UNARY_MATHS_OP(cos); _XVEC_UNARY_MATHS_OP(acos);
_XVEC_UNARY_MATHS_OP(tan); _XVEC_UNARY_MATHS_OP(atan);

_XVEC_UNARY_MATHS_OP(sinh); _XVEC_UNARY_MATHS_OP(asinh);
_XVEC_UNARY_MATHS_OP(cosh); _XVEC_UNARY_MATHS_OP(acosh);
_XVEC_UNARY_MATHS_OP(tanh); _XVEC_UNARY_MATHS_OP(atanh);
// atan2

// Power
_XVEC_UNARY_MATHS_OP(cbrt);
_XVEC_BINARY_MATHS_OP(hypot);
_XVEC_BINARY_MATHS_OP(pow);

// Error and gamma (lgamma not supported yet in SVML)
_XVEC_UNARY_MATHS_OP(erf);
_XVEC_UNARY_MATHS_OP(erfc);

/// @} cpp26_math

} // namespace _XVEC_NAMESPACE::simd

namespace _XVEC_NAMESPACE
{
  using simd::abs;
  using simd::copysign;
  using simd::fma;
  using simd::sqrt;
  using simd::ceil;
  using simd::floor;
  using simd::trunc;
  using simd::rint;
  using simd::round;

  using simd::isinf;
  using simd::isnan;
  using simd::isfinite;
  using simd::isnormal;
  using simd::signbit;
  using simd::isgreater;
  using simd::isgreaterequal;
  using simd::isless;
  using simd::islessequal;
  using simd::islessgreater;
  using simd::isunordered;

  using simd::exp;
  using simd::exp2;
  using simd::expm1;
  using simd::log;
  using simd::log2;
  using simd::log1p;

  using simd::sin;
  using simd::asin;
  using simd::cos;
  using simd::acos;
  using simd::tan;
  using simd::atan;
  using simd::sinh;
  using simd::asinh;
  using simd::cosh;
  using simd::acosh;
  using simd::tanh;
  using simd::atanh;

  using simd::cbrt;
  using simd::hypot;
  using simd::pow;

  using simd::erf;
  using simd::erfc;
}

// The maths-op generator macros exist only to stamp out the declarations above,
// so retire them here rather than leaking them into every translation unit that
// includes <xvec/simd>. Undefining a macro that was never defined is well-formed,
// so the target-specific helpers need no guards. Note this must happen after the
// last expansion above, since the target-specific helpers are expanded lazily
// inside _XVEC_UNARY_MATHS_OP / _XVEC_BINARY_MATHS_OP.
#undef _XVEC_UNARY_MATHS_OP
#undef _XVEC_BINARY_MATHS_OP

#undef _XVEC_SSE_UNARY_MATHS_OP
#undef _XVEC_SSE_BINARY_MATHS_OP
#undef _XVEC_AVX_UNARY_MATHS_OP
#undef _XVEC_AVX_BINARY_MATHS_OP
#undef _XVEC_AVX512_UNARY_MATHS_OP
#undef _XVEC_AVX512_BINARY_MATHS_OP
#undef _XVEC_AVX512FP16_UNARY_MATHS_OP
#undef _XVEC_AVX512FP16_BINARY_MATHS_OP