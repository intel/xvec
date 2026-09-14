//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <cstdint>
#include <cstddef>
#include <concepts>
#include <limits> // std::numeric_limits

namespace _XVEC_NAMESPACE::simd
{

/// \defgroup simd_bit bit header overloads for simd
/// @brief SIMD versions of functions which appear in C++20 bit header.

/// @brief Compute the result of rotating each vec element left by a given
/// offset. Also known as circular shift. Values rotated off the end will appear
/// at the other end. If the shift value is negative it is equivalent to
/// shifting right by the negated shift.
/// @{
/// @ingroup simd_bit
/// @tparam _Up The unsigned vec type to rotate.
/// @tparam _Sp The offset vec type by which to rotate.
/// @param u The value to rotate
/// @param s The offset by which to rotate
/// @return A new simd where every element is the circular shift by s of the
/// corresponding element of the simd value.
template<vec_unsigned_integer _Up, vec_integral _Sp>
  requires (_Up::size() == _Sp::size() &&
            sizeof(typename _Up::value_type) == sizeof(typename _Sp::value_type))
constexpr _Up rotl(const _Up& u, const _Sp& s)
{
  if (std::is_constant_evaluated())
    return _Up([=](auto i) { return typename _Up::value_type(std::rotl(u[i], s[i])); });
  else
    return detail::rotate_left(target, u, s);
}

template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up rotl(const _Up& u, _Sp s) {
  using _Rp = std::make_signed_t<typename _Up::value_type>;
  return rotl(u, rebind_t<_Rp, _Up>(_Rp(s))); }
///@}

///@{
/// @brief Compute the result of rotating each basic_vec element right by a given
/// offset. Also known as circular shift. Values rotated off the end will appear
/// at the other end. If the shift value is negative it is equivalent to
/// shifting left by the negated shift.
/// @ingroup simd_bit
/// @tparam _Up The unsigned vec type to rotate.
/// @tparam _Sp The offset vec type by which to rotate.
/// @param u The value to rotate
/// @param s The offset by which to rotate
/// @return A new basic_vec where every element is the circular shift by s of the
/// corresponding element of the basic_vec value.
template<vec_unsigned_integer _Up, vec_integral _Sp>
  requires (_Up::size() == _Sp::size() &&
            sizeof(typename _Up::value_type) == sizeof(typename _Sp::value_type))
constexpr _Up rotr(const _Up& u, const _Sp& s)
{
  if (std::is_constant_evaluated())
    return _Up([=](auto i) { return typename _Up::value_type(std::rotr(u[i], s[i])); });
  else
    return detail::rotate_right(target, u, s);
}

template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up rotr(const _Up& u, _Sp s) {
  using _Rp = std::make_signed_t<typename _Up::value_type>;
  return rotr(u, rebind_t<_Rp, _Up>(_Rp(s))); }
/// @}

/// @brief Return the number of consecutive 0 bits in each element starting from the most significant bit.
/// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param v The input value
/// @return A basic_vec containing the number of consecutive 0 bits in each respective element.
template<vec_unsigned_integer _Vp>
constexpr auto countl_zero(const _Vp& v) noexcept {
  using _Rp = std::make_signed_t<typename _Vp::value_type>;
  if (std::is_constant_evaluated())
    return rebind_t<_Rp, _Vp>([=](auto i){ return _Rp(std::countl_zero(v[i])); });
  else
    return rebind_t<_Rp, _Vp>(detail::clz(target, v));
}

/// @brief Return the number of consecutive 1 bits in each element starting from the most significant bit.
/// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param x The input value
/// @return A basic_vec containing the number of consecutive 1 bits in each respective element.
template<vec_unsigned_integer _Vp> constexpr auto countl_one(const _Vp& x) noexcept { return countl_zero(~x); }

/// @brief Return the number of consecutive 0 bits in each element starting from the least significant bit.
/// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param v The input value
/// @return A basic_vec containing the number of consecutive 0 bits starting from the LSB in each respective element.
template<vec_unsigned_integer _Vp>
constexpr auto countr_zero(const _Vp& v) noexcept {
  using _Rp = std::make_signed_t<typename _Vp::value_type>;
  if (std::is_constant_evaluated())
    return rebind_t<_Rp, _Vp>([&](auto i){ return _Rp(std::countr_zero(v[i])); });
  else
  {
    // Remove all bits except those to right of the lowest set bit
    // http://0x80.pl/notesen/2023-01-31-avx512-bsf.html
    auto lsbBits = ~v & (v - detail::cw<1>);
    return rebind_t<_Rp, _Vp>(detail::popcount(target, lsbBits));
  }
}

/// @brief Return the number of consecutive 1 bits in each element starting from the least significant bit.
/// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param x The input value
/// @return A basic_vec containing the number of consecutive 1 bits from the LSB in each respective element.
template<vec_unsigned_integer _Vp> constexpr auto countr_one(const _Vp& x) noexcept { return countr_zero(~x); }

/// @brief Count the number of 1 bits in each basic_vec element
// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param x The input value
/// @return A basic_vec containing the total number of 1 bits in each respective input element
template<vec_unsigned_integer _Vp>
constexpr auto popcount(const _Vp& x) noexcept {
  using _Rp = std::make_signed_t<typename _Vp::value_type>;
  if (std::is_constant_evaluated())
    return rebind_t<_Rp, _Vp>([=](auto i) { return _Rp(std::popcount(x[i])); });
  else
    return rebind_t<_Rp, _Vp>(detail::popcount(target, x));
}

/// @brief Check if each element is an integral power-of-2.
// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param x The input value
/// @return A mask where each element is set to true if exactly one bit is used in the
/// respective element, or false otherwise.
template<vec_unsigned_integer _Vp>
constexpr typename _Vp::mask_type has_single_bit(const _Vp& x) noexcept {
  return detail::has_single_bit(target, x);
}

/// @brief Compute the bit width of each element (i.e., how many bits are needed to store the element value).
// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param x The input value
/// @return A signed basic_vec containing the width of each respective element
template<vec_unsigned_integer _Vp>
constexpr rebind_t<std::make_signed_t<typename _Vp::value_type>, _Vp>
bit_width(const _Vp& x) noexcept {
  using _R = rebind_t<std::make_signed_t<typename _Vp::value_type>, _Vp>;
  return _R(detail::cw<std::numeric_limits<typename _Vp::value_type>::digits> - countl_zero(x));
}

/// @brief For every element compute the smallest integral power of two that is
/// not smaller than the element's value.
// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param x The input value
/// @return A signed basic_vec containing the result of std::bit_ceil for that element.
template<vec_unsigned_integer _Vp>
constexpr _Vp bit_ceil(const _Vp& x) noexcept {
  if (std::is_constant_evaluated())
    return _Vp([=](auto i) { return std::bit_ceil(x[i]); });
  else
  {
    // Normally bit_ceil would need to be wary of shifts, since some types get
    // promoted and would give the wrong result. This is't a problem for SIMD
    // operations since promotion doesn't occur.
    constexpr auto one = _Vp(uint8_t(1));
    auto shift_amount = _Vp(bit_width(x - one));
    return select(x < uint8_t(2), one, one << shift_amount);
  }
}

/// @brief For every element compute the largest integral power of two that is not greater than the element's value.
// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param v The input value
/// @return A signed basic_vec containing the result of std::bit_floor for that element.
template<vec_unsigned_integer _Vp>
constexpr _Vp bit_floor(const _Vp& v) noexcept {
  using _Tp = typename _Vp::value_type;
  if (std::is_constant_evaluated())
    return _Vp([=](auto i) { return std::bit_floor(v[i]); });
  else
  {
    auto log2 = rebind_cast<_Tp>(detail::cw<std::numeric_limits<_Tp>::digits - 1> - countl_zero(v));
    return select(v != _Vp(), uint8_t(1) << log2, _Vp());
  }
}

/// @brief Bit reverse is now part of C++29. It used to be called bitswap in older releases of this library.
/// \ingroup simd_bit
/// @tparam _Vp The type of the vec input.
/// @param x
/// @return Each element will have its bits reversed.
template<vec_unsigned_integer _Vp>
constexpr _Vp bit_reverse(const _Vp& x) noexcept {
  if (std::is_constant_evaluated())
    return detail::bitreverse(generic_tag{}, x);
  else
    return detail::bitreverse(target, x); }

/// @brief Reverses the bytes in each integer basic_vec element. Byte swap is part of C++23 but is useful enough to include it before then.
/// \ingroup simd_bit
/// @tparam _Vp the type of the vec input.
/// @param v
/// @return Each element will have its bytes reversed.
template<vec_integral _Vp>
constexpr _Vp byteswap(const _Vp& v) noexcept {
  if (std::is_constant_evaluated())
  {
    // byteswap only exists in C++23, so until then make our own.
    auto bswap = [](auto b) {
      auto value_representation = std::bit_cast<std::array<std::byte, sizeof(b)>>(b);
      std::ranges::reverse(value_representation);
      return std::bit_cast<decltype(b)>(value_representation);
    };
    return _Vp([=](auto i) { return bswap(v[i]); });
  }

  return detail::byteswap(target, v);
}

///@{
/// @brief Compute the result of a funnel shift right of two basic_vec values
/// by a given offset. Conceptually concatenates @p high and @p low into a
/// value twice as wide, shifts right by @p s, and returns the low N bits.
/// @ingroup simd_bit
/// @tparam _Up The unsigned vec type of the values to shift.
/// @tparam _Sp The offset vec type by which to shift.
/// @param high The value providing the upper bits of the concatenation.
/// @param low The value providing the lower bits of the concatenation.
/// @param s The offset by which to shift. Precondition: 0 <= s[i] < N for all i.
/// @return A new basic_vec where every element is the funnel shift right of the
/// corresponding elements of @p high and @p low by the corresponding element of @p s.
template<vec_unsigned_integer _Up, vec_integral _Sp>
  requires (sizeof(typename _Up::value_type) == sizeof(typename _Sp::value_type))
constexpr _Up funnel_shift_right(const _Up& high, const _Up& low, const _Sp& s)
{
  if (std::is_constant_evaluated())
    return _Up([=](auto i) { return typename _Up::value_type(detail::funnel_shift_right(high[i], low[i], s[i])); });
  else
    return detail::fsr(target, high, low, s);
}

template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up funnel_shift_right(const _Up& high, const _Up& low, _Sp s)
  { return funnel_shift_right(high, low, _Up(typename _Up::value_type(s))); }
/// @}

///@{
/// @brief Compute the result of a funnel shift left of two basic_vec values
/// by a given offset. Conceptually concatenates @p high and @p low into a
/// value twice as wide, shifts left by @p s, and returns the high N bits.
/// @ingroup simd_bit
/// @tparam _Up The unsigned vec type of the values to shift.
/// @tparam _Sp The offset vec type by which to shift.
/// @param high The value providing the upper bits of the concatenation.
/// @param low The value providing the lower bits of the concatenation.
/// @param s The offset by which to shift. Precondition: 0 <= s[i] < N for all i.
/// @return A new basic_vec where every element is the funnel shift left of the
/// corresponding elements of @p high and @p low by the corresponding element of @p s.
template<vec_unsigned_integer _Up, vec_integral _Sp>
  requires (sizeof(typename _Up::value_type) == sizeof(typename _Sp::value_type))
constexpr _Up funnel_shift_left(const _Up& high, const _Up& low, const _Sp& s)
{
  if (std::is_constant_evaluated())
    return _Up([=](auto i) { return typename _Up::value_type(detail::funnel_shift_left(high[i], low[i], s[i])); });
  else
    return detail::fsl(target, high, low, s);
}

template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up funnel_shift_left(const _Up& high, const _Up& low, _Sp s)
  { return funnel_shift_left(high, low, _Up(typename _Up::value_type(s))); }
/// @}

} // namespace _XVEC_NAMESPACE::simd

namespace _XVEC_NAMESPACE
{
  using simd::byteswap;
  using simd::bit_floor;
  using simd::bit_ceil;
  using simd::has_single_bit;
  using simd::rotl;
  using simd::rotr;
  using simd::bit_width;
  using simd::countl_zero;
  using simd::countl_one;
  using simd::countr_zero;
  using simd::countr_one;
  using simd::popcount;
  using simd::funnel_shift_left;
  using simd::funnel_shift_right;
}
