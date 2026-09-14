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

/// \defgroup simd_math simd C++ math header overloads
/// @brief Support and overloads for simd versions of C++ functions from the math header.

/// @brief Compute the absolute value of each input vec element. Note that handling abs of an integral is an xvec extension.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The vec value
/// @return A simd vec value in which each element is the absolute value of the respective input vec element.
///@{
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> abs(const _Vp& v) { return detail::abs(target, v); }
template<vec_signed_integral _Vp> constexpr _Vp abs(const _Vp& v) { return detail::abs(target, v); }
///@}

/// @brief Return a value with the magnitude of the first parameter and the sign of the second.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec
/// @param magnitude The magnitude which will be copied to the output
/// @param sign The sign which will be copied to the output
/// @return A value with the magnitude of the first parameter and the sign of the second.
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

/// @brief Compute the approximate reciprocal of the input.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec
/// @param v The basic_vec value from which to compute the approximate reciprocal.
/// @return A basic_vec value where every element is the approximate reciprocal of the corresponding input basic_vec element.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> rcp (const _Vp& v) { return detail::rcp(target, v); }

/// @brief Compute the approximate reciprocal square-root of the input.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec
/// @param v The vec value from which to compute the approximate reciprocal square root
/// @return A vec value where every element is the approximate reciprocal
/// square root of the corresponding input vec element.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> rsqrt(const _Vp& v) { return detail::rsqrt(target, v); }

/// @brief Compute the square-root of the input to within 0.5ULP as guaranteed by IEEE floating-point.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The basic_vec value from which to compute the square root
/// @return A basic_vec value where every element is the square root of the corresponding input basic_vec element.
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp> sqrt (const _Vp& v) { return detail::sqrt(target, v); }

/// @brief Compute (lhs * rhs) * acc as if to infinite precision, and rounded
/// only once to fit the result type. The complex overload is an xvec extension.
/// \ingroup simd_math
/// \tparam _Vp The type of the basic_vec
/// \param lhs The left hand value in the multiply.
/// \param rhs The right hand value in the multiply.
/// \param acc The accumulator to which the result of the multiply should be added.
/// \return A new basic_vec value in which each element is formed by applying the fma to each respective element of the inputs.
///@{
template<detail::math_floating_point _Vp> constexpr detail::deduced_vec_t<_Vp>
  fma (const _Vp& lhs, const _Vp& rhs, const _Vp& acc) { return detail::fma(target, lhs, rhs, acc); }

template<vec_complex _Vp>
constexpr _Vp fma (const _Vp& lhs, const _Vp& rhs, const _Vp& acc)
  { return detail::fma(target, lhs, rhs, acc); }
///@}

/// @brief For all even-indexed elements in the vec, compute (lhs * rhs - acc) and for
/// all odd-indexed elements compute (lhs * rhs + acc). The expressions are
/// computed as if to infinite precision, and rounded only once to fit the
/// result type. This is an xvec extension.
/// \ingroup simd_math
/// \tparam _Vp The type of the basic_vec.
/// \param lhs The left hand value in the multiply.
/// \param rhs The right hand value in the multiply.
/// \param acc The accumulator to which the result of the multiply should be alternatively subtracted or added.
/// \return A new basic_vec value in which each element is formed by applying (lhs * rhs) +- acc to each respective element of the inputs.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> fmaddsub(const _Vp& lhs, const _Vp& rhs, const _Vp& acc)
  { return detail::fmaddsub(target, lhs, rhs, acc); }

/// @brief For all even-indexed elements in the basic_vec compute (lhs * rhs + acc) and for
/// all odd-indexed elements compute (lhs * rhs - acc). The expressions are
/// computed as if to infinite precision, and rounded only once to fit the
/// result type. This is an xvec extension.
/// \ingroup simd_math
/// \tparam _Vp The type of the basic_vec
/// \param lhs The left hand value in the multiply.
/// \param rhs The right hand value in the multiply.
/// \param acc The accumulator to which the result of the multiply should be alternatively subtracted or added.
/// \return A new basic_vec value in which each element is formed by applying (lhs * rhs) +- acc to each respective element of the inputs.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> fmsubadd(const _Vp& lhs, const _Vp& rhs, const _Vp& acc)
  { return detail::fmsubadd(target, lhs, rhs, acc); }

/// @brief For each element compute the smallest integer not less than the given value.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The basic_vec value
/// @return A basic_vec value where each element is the smallest integer not less than the given element value.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> ceil(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_ceil)
    return __builtin_elementwise_ceil(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::CEIL>(target, v);
  #endif
}

/// @brief For each element compute the largest integer no greater than the given value.
/// \ingroup simd_math
/// @tparam _Vp The type of the basic_vec
/// @param v The vec value
/// @return A vec value where each element is the largest integer no greater than the given element value.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> floor(const _Vp& v)
{
  #if __has_builtin(__builtin_elementwise_floor)
    return __builtin_elementwise_floor(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::FLOOR>(target, v);
  #endif
}

/// @brief For each element compute the nearest integer not greater than the magnitude of the element value.
/// \ingroup simd_math
/// @tparam _Vp The type of the basic_vec
/// @param v The basic_vec value
/// @return A basic_vec value where each element is the nearest integer not greater than the magnitude of the input element.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> trunc(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_trunc)
    return __builtin_elementwise_trunc(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::TRUNC>(target, v);
  #endif
}

/// \ingroup simd_math
/// @brief For each element compute the nearest integer value using the current rounding mode.
/// Unlike `round`, this function respects the floating-point environment's current
/// rounding mode.
/// @tparam _Vp The type of the basic_vec
/// @param v The basic_vec value
/// @return A basic_vec value where each element is the nearest integer using the current rounding mode.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> rint(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_rint)
    return __builtin_elementwise_rint(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::RINT>(target, v);
  #endif
}

/// @brief For each element compute the nearest integer value
/// \ingroup simd_math
/// @tparam _Vp
/// @param v The basic_vec value
/// @return A basic_vec value where each element is the nearest integer.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_vec_t<_Vp> round(const _Vp& v) {
  #if __has_builtin(__builtin_elementwise_roundeven)
    return __builtin_elementwise_round(v.to_builtin());
  #else
    return detail::round_op<detail::RoundOp::ROUND>(target, v);
  #endif
}

namespace detail
{
///@{
/// Generic implementation of the various classification functions.
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
  // ones, and the fraction all zeros. Checking a NaN against itself is another
  // way but is fragile with respect to compiler optimisations.
  const auto asC = vec_as_container(v);
  return ((asC & expMask) == expMask) & ((asC & fractionMask) != uint8_t(0));
}

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isinf(generic_tag tag, const _Vp& v)
  { return !isnan(tag, v) && (abs(v) > std::numeric_limits<_Vp>::max()); }

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isfinite(generic_tag tag, const _Vp& v)
  { return !(isinf(tag, v) || isnan(tag, v)); }

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isnormal(generic_tag, const _Vp& v)
{
  return (abs(v) >= std::numeric_limits<_Vp>::min()) &&
         (abs(v) <= std::numeric_limits<_Vp>::max());
}
///@}

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> signbit(generic_tag, const _Vp& v)
{
  constexpr auto signBit = vec_as_container(_Vp(typename _Vp::value_type(-0.0)));
  auto m = (vec_as_container(v) & signBit) != uint8_t(0);
  return detail::deduced_mask_t<_Vp>(m);
}

}

// namespace detail

/// @brief Compute a mask which indicates which elements in a vec value represent positive or negative infinities.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The basic_vec value to query
/// @return A basic_vec mask where each respective bit of the mask will be set if the
/// corresponding input element is positive or negative infinity, or cleared
/// otherwise.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isinf(const _Vp& v) { return detail::isinf(target, v); }

/// @brief Compute a basic_vec mask which indicates which elements in a basic_vec value represent a not-a-number floating-point value.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The basic_vec value to query
/// @return A mask where each respective bit of the mask will be set if the
/// corresponding input element is not-a-number, or cleared otherwise.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isnan(const _Vp& v) { return detail::isnan(target, v); }

/// @brief Compute a mask which indicates which elements in a vec value represent
/// finite values. Values which are NaN or infinite will not be finite, and all
/// other floating-point values will be.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The simd value to query
/// @return A simd mask where each respective bit of the mask will be set if the
/// corresponding input element is a finite floating-point value or cleared otherwise.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isfinite(const _Vp& v) { return detail::isfinite(target, v); }

/// @brief Compute a mask which indicates which elements in a vec value
/// represents a normal floating point value, which isn't NaN, zero, sub-normal
/// or infinite.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The basic_vec value to query
/// @return A basic_vec mask where each respective bit of the mask will be set if the
/// corresponding input element is a normal number or cleared otherwise.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isnormal(const _Vp& v) { return detail::isnormal(target, v); }

/// @brief Compute a basic_vec mask which indicates which elements in a basic_vec value represent a negative value.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The basic_vec value to query
/// @return A mask where each respective bit of the mask will be set if the
/// corresponding input element is negative, or cleared otherwise.
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> signbit(const _Vp& v) {  return detail::signbit(target, v); }

/// @brief Compute a basic_vec mask which indicates which elements in a
/// basic_vec value represent a negative value. This is an integral version of
/// the signbit function which doesn't exist in C++26, but is an extension in
/// xvec.
/// @ingroup simd_math
/// @tparam _Vp The type of the basic_vec.
/// @param v The basic_vec value to query
/// @return A mask where each respective bit of the mask will be set if the
/// corresponding input element is negative, or cleared otherwise.
template<vec_integral _Vp>
constexpr typename _Vp::mask_type signbit(const _Vp& v) {
  if constexpr (vec_unsigned_integral<_Vp>)
    return typename _Vp::mask_type(false); // Unsigned integer can never become signed
  else
    return v < _Vp(); // Signed integer 
}

/// @brief Comparison functions which produce a mask result. These are the
/// quiet (non-signalling) comparison functions from [simd.math]. They are
/// specified as quiet comparisons (no FE_INVALID raised for unordered
/// operands), and for NaN operands the corresponding result bit is false. For
/// this implementation each function is a thin wrapper over the equivalent
/// relational operator, which already produces a mask and already gives the
/// correct (false) result for NaN-involving comparisons. Only the all-vector
/// overload is provided; the mixed scalar/vector overloads from the standard
/// are not implemented yet.
///@{
template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isgreater(const _Vp& x, const _Vp& y) { return x > y; }

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isgreaterequal(const _Vp& x, const _Vp& y) { return x >= y; }

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isless(const _Vp& x, const _Vp& y) { return x < y; }

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> islessequal(const _Vp& x, const _Vp& y) { return x <= y; }

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> islessgreater(const _Vp& x, const _Vp& y) { return (x < y) | (x > y); }

template<detail::math_floating_point _Vp>
constexpr detail::deduced_mask_t<_Vp> isunordered(const _Vp& x, const _Vp& y) { return isnan(x) | isnan(y); }
///@}

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
