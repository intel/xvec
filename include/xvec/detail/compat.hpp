//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

#include <xvec/detail/config.hpp>

/// Detect a scoped enum. This will be provided in C++23 onwards.
#if defined(__cpp_lib_is_scoped_enum) && (__cpp_lib_is_scoped_enum >= 202011L)
// C++23 library feature-test macro for the std::is_scoped_enum trait
namespace _XVEC_NAMESPACE::simd::detail {
  template<class E>
  concept is_scoped_enum = std::is_scoped_enum_v<E>;
}
#else
namespace _XVEC_NAMESPACE::simd::detail {
  template<class E>
  concept is_scoped_enum =
    std::is_enum_v<E> &&
    (!std::is_convertible_v<E, std::underlying_type_t<E>>);
}
#endif

namespace _XVEC_NAMESPACE::simd::detail {

template<class T>
concept saturation_integer =
  std::integral<T> &&
  !std::same_as<std::remove_cv_t<T>, bool> &&
  !std::same_as<std::remove_cv_t<T>, char> &&
  !std::same_as<std::remove_cv_t<T>, wchar_t> &&
  !std::same_as<std::remove_cv_t<T>, char8_t> &&
  !std::same_as<std::remove_cv_t<T>, char16_t> &&
  !std::same_as<std::remove_cv_t<T>, char32_t>;

/// Convert an integer to another integer type, clamping values outside the
/// destination type's representable range.
template<saturation_integer To, saturation_integer From>
[[nodiscard]] constexpr To saturating_cast(From value) noexcept
{
  if (std::in_range<To>(value))
    return static_cast<To>(value);

  // A negative out-of-range value lies below the destination range. This also
  // covers conversion to an unsigned destination, whose minimum is zero.
  if constexpr (std::signed_integral<From>) {
    if (value < 0)
      return std::numeric_limits<To>::min();
  }

  // Every remaining out-of-range value lies above the destination range.
  return std::numeric_limits<To>::max();
}

/// Add two integers, clamping an unrepresentable mathematical result.
template<saturation_integer T>
[[nodiscard]] constexpr T saturating_add(T lhs, T rhs) noexcept
{
  T result{};

  if (!__builtin_add_overflow(lhs, rhs, &result))
    return result;

  constexpr T lowest = std::numeric_limits<T>::min();
  constexpr T highest = std::numeric_limits<T>::max();

  if constexpr (std::unsigned_integral<T>) {
    // Unsigned addition can overflow only above the upper endpoint.
    return highest;
  } else {
    // Signed addition can overflow only when both operands have the same
    // sign. Their shared sign therefore determines the saturation direction.
    return rhs < 0 ? lowest : highest;
  }
}

/// Subtract two integers, clamping an unrepresentable mathematical result.
template<saturation_integer T>
[[nodiscard]] constexpr T saturating_sub(T lhs, T rhs) noexcept
{
  T result{};

  if (!__builtin_sub_overflow(lhs, rhs, &result))
    return result;

  constexpr T lowest = std::numeric_limits<T>::min();
  constexpr T highest = std::numeric_limits<T>::max();

  if constexpr (std::unsigned_integral<T>) {
    // Unsigned subtraction is unrepresentable only when the mathematical
    // result falls below zero.
    return lowest;
  } else {
    // Subtracting a negative value can overflow only toward the upper
    // endpoint. Subtracting a positive value can overflow only toward the
    // lower endpoint.
    return rhs < 0 ? highest : lowest;
  }
}

/// Multiply two integers, clamping an unrepresentable mathematical result.
template<saturation_integer T>
[[nodiscard]] constexpr T saturating_mul(T lhs, T rhs) noexcept
{
  T result{};

  if (!__builtin_mul_overflow(lhs, rhs, &result))
    return result;

  constexpr T lowest = std::numeric_limits<T>::min();
  constexpr T highest = std::numeric_limits<T>::max();

  if constexpr (std::unsigned_integral<T>) {
    // An unsigned mathematical product cannot be negative, so overflow is
    // always above the upper endpoint.
    return highest;
  } else {
    // Operands with equal signs have a positive mathematical product and
    // therefore overflow toward the upper endpoint. Different signs produce
    // a negative product and overflow toward the lower endpoint.
    return (lhs < 0) == (rhs < 0) ? highest : lowest;
  }
}

/// Divide two integers, clamping an unrepresentable mathematical result.
template<saturation_integer T>
[[nodiscard]] constexpr T saturating_div(T lhs, T rhs) noexcept
{
  if constexpr (std::signed_integral<T>) {
    // lowest / -1 is the only integral division with a nonzero divisor whose
    // mathematical result is not representable in the operand type.
    if (lhs == std::numeric_limits<T>::min() && rhs == T{-1})
      return std::numeric_limits<T>::max();
  }

  return lhs / rhs;
}

// Transparent binary function objects for bitwise shift operations. These are
// used as the default shift operators for simd types.
template<typename T = void>
struct bit_lshift {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return lhs << rhs;
  }
};

template<typename T = void>
struct bit_rshift {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return lhs >> rhs;
  }
};

// Transparent specializations
template<>
struct bit_lshift<void> {
  template<typename T, typename U>
  constexpr auto operator()(T&& lhs, U&& rhs) const
    -> decltype(std::forward<T>(lhs) << std::forward<U>(rhs)) {
    return std::forward<T>(lhs) << std::forward<U>(rhs);
  }
  
  using is_transparent = void;
};

template<>
struct bit_rshift<void> {
  template<typename T, typename U>
  constexpr auto operator()(T&& lhs, U&& rhs) const
    -> decltype(std::forward<T>(lhs) >> std::forward<U>(rhs)) {
    return std::forward<T>(lhs) >> std::forward<U>(rhs);
  }
  
  using is_transparent = void;
};

template<typename _Tp>
concept funnel_shift_source =
  std::same_as<std::remove_cvref_t<_Tp>, unsigned char> ||
  std::same_as<std::remove_cvref_t<_Tp>, unsigned short> ||
  std::same_as<std::remove_cvref_t<_Tp>, unsigned int> ||
  std::same_as<std::remove_cvref_t<_Tp>, unsigned long> ||
  std::same_as<std::remove_cvref_t<_Tp>, unsigned long long>;
  
/// funnel_shift_right(high, low, s)
///
/// Preconditions: 0 <= s < N, where N = numeric_limits<T>::digits.
/// Returns: (low >> r) | (high << ((N - r) % N)), where r = static_cast<unsigned>(s).
/// Remark: funnel_shift_right(high, low, 0) returns low.
template <funnel_shift_source T, std::integral S>
[[nodiscard]] constexpr T funnel_shift_right(T high, T low, S s) noexcept {
    constexpr auto N = std::numeric_limits<T>::digits;
    assert(s >= 0 && static_cast<unsigned>(s) < static_cast<unsigned>(N));
    const auto r = static_cast<unsigned>(s);
    if (r == 0) return low;
    return static_cast<T>((low >> r) | (high << (N - r)));
}

/// funnel_shift_left(high, low, s)
///
/// Preconditions: 0 <= s < N, where N = numeric_limits<T>::digits.
/// Returns: (high << r) | (low >> ((N - r) % N)), where r = static_cast<unsigned>(s).
/// Remark: funnel_shift_left(high, low, 0) returns high.
template <funnel_shift_source T, std::integral S>
[[nodiscard]] constexpr T funnel_shift_left(T high, T low, S s) noexcept {
    constexpr auto N = std::numeric_limits<T>::digits;
    assert(s >= 0 && static_cast<unsigned>(s) < static_cast<unsigned>(N));
    const auto r = static_cast<unsigned>(s);
    if (r == 0) return high;
    return static_cast<T>((high << r) | (low >> (N - r)));
}

} // namespace _XVEC_NAMESPACE::simd::detail

#if defined(__FLT16_MIN__) && !defined(__STDCPP_FLOAT16_T__)
// The compiler has _Float16 as an extension, but the standard library has
// not yet provided std::numeric_limits<_Float16>, so supply our own.

namespace std {
template <>
struct numeric_limits<_Float16> {
  static constexpr bool is_specialized = true;

  static constexpr int  digits         = 11;
  static constexpr int  digits10       = 3;
  static constexpr int  max_digits10   = 5;
  static constexpr bool is_signed      = true;
  static constexpr bool is_integer     = false;
  static constexpr bool is_exact       = false;
  static constexpr int  radix          = 2;
  static constexpr int  min_exponent   = -13;
  static constexpr int  min_exponent10 = -4;
  static constexpr int  max_exponent   = 16;
  static constexpr int  max_exponent10 = 4;

  static constexpr bool has_infinity      = true;
  static constexpr bool has_quiet_NaN     = true;
  static constexpr bool has_signaling_NaN = true;

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool              traps           = false;
  static constexpr bool              tinyness_before = false;
  static constexpr float_round_style round_style     = round_to_nearest;

  static constexpr _Float16 min()           noexcept { return static_cast<_Float16>(__FLT16_MIN__); }
  static constexpr _Float16 max()           noexcept { return static_cast<_Float16>(__FLT16_MAX__); }
  static constexpr _Float16 lowest()        noexcept { return static_cast<_Float16>(-__FLT16_MAX__); }
  static constexpr _Float16 epsilon()       noexcept { return static_cast<_Float16>(__FLT16_EPSILON__); }
  static constexpr _Float16 round_error()   noexcept { return static_cast<_Float16>(0.5f); }
  static constexpr _Float16 infinity()      noexcept { return static_cast<_Float16>(__builtin_huge_valf()); }
  static constexpr _Float16 quiet_NaN()     noexcept { return static_cast<_Float16>(__builtin_nanf("")); }
  static constexpr _Float16 signaling_NaN() noexcept { return static_cast<_Float16>(__builtin_nansf("")); }
  static constexpr _Float16 denorm_min()    noexcept { return static_cast<_Float16>(__FLT16_DENORM_MIN__); }
};

} // namespace std
#endif // defined(__FLT16_MIN__) && !defined(__STDCPP_FLOAT16_T__)
