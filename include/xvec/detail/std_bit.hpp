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

/**
 * @defgroup cpp26_bit C++26: Bit manipulation
 * @brief Element-wise bit counting, rotation, power-of-two, and byte-order operations.
 *
 * @see https://eel.is/c++draft/simd.bit
 * @{
 */

/// @brief Rotate each unsigned element left by its corresponding offset.
/// Bits rotated off one end reappear at the other end.
/// Negative offsets rotate in the opposite direction.
/// @tparam _Up The unsigned vector type.
/// @tparam _Sp The integral offset vector type.
/// @param u The input vector.
/// @param s The element-wise rotation offsets.
/// @return A vector containing the rotated elements.
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

/// @brief Rotate every unsigned element left using a shared offset.
/// @param u The input vector.
/// @param s The shared rotation offset.
/// @return A vector containing the rotated elements.
template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up rotl(const _Up& u, _Sp s) {
  using _Rp = std::make_signed_t<typename _Up::value_type>;
  return rotl(u, rebind_t<_Rp, _Up>(_Rp(s))); }

/// @brief Rotate each unsigned element right by its corresponding offset.
/// Bits rotated off one end reappear at the other end.
/// Negative offsets rotate in the opposite direction.
/// @tparam _Up The unsigned vector type.
/// @tparam _Sp The integral offset vector type.
/// @param u The input vector.
/// @param s The element-wise rotation offsets.
/// @return A vector containing the rotated elements.
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

/// @brief Rotate every unsigned element right using a shared offset.
/// @param u The input vector.
/// @param s The shared rotation offset.
/// @return A vector containing the rotated elements.
template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up rotr(const _Up& u, _Sp s) {
  using _Rp = std::make_signed_t<typename _Up::value_type>;
  return rotr(u, rebind_t<_Rp, _Up>(_Rp(s))); }

/// @brief Count consecutive zero bits from the most significant bit of each element.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A signed integral vector containing the counts.
template<vec_unsigned_integer _Vp>
constexpr auto countl_zero(const _Vp& v) noexcept {
  using _Rp = std::make_signed_t<typename _Vp::value_type>;
  if (std::is_constant_evaluated())
    return rebind_t<_Rp, _Vp>([=](auto i){ return _Rp(std::countl_zero(v[i])); });
  else
    return rebind_t<_Rp, _Vp>(detail::clz(target, v));
}

/// @brief Count consecutive one bits from the most significant bit of each element.
/// @tparam _Vp The input vector type.
/// @param x The input vector.
/// @return A signed integral vector containing the counts.
template<vec_unsigned_integer _Vp> constexpr auto countl_one(const _Vp& x) noexcept { return countl_zero(~x); }

/// @brief Count consecutive zero bits from the least significant bit of each element.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A signed integral vector containing the counts.
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

/// @brief Count consecutive one bits from the least significant bit of each element.
/// @tparam _Vp The input vector type.
/// @param x The input vector.
/// @return A signed integral vector containing the counts.
template<vec_unsigned_integer _Vp> constexpr auto countr_one(const _Vp& x) noexcept { return countr_zero(~x); }

/// @brief Count the one bits in each element.
/// @tparam _Vp The input vector type.
/// @param x The input vector.
/// @return A signed integral vector containing the population counts.
template<vec_unsigned_integer _Vp>
constexpr auto popcount(const _Vp& x) noexcept {
  using _Rp = std::make_signed_t<typename _Vp::value_type>;
  if (std::is_constant_evaluated())
    return rebind_t<_Rp, _Vp>([=](auto i) { return _Rp(std::popcount(x[i])); });
  else
    return rebind_t<_Rp, _Vp>(detail::popcount(target, x));
}

/// @brief Identify elements that are integral powers of two.
/// @tparam _Vp The input vector type.
/// @param x The input vector.
/// @return A mask that is true where the corresponding element contains exactly one set bit.
template<vec_unsigned_integer _Vp>
constexpr typename _Vp::mask_type has_single_bit(const _Vp& x) noexcept {
  return detail::has_single_bit(target, x);
}

/// @brief Compute the number of bits needed to represent each element.
/// @tparam _Vp The input vector type.
/// @param x The input vector.
/// @return A signed integral vector containing the bit widths.
template<vec_unsigned_integer _Vp>
constexpr rebind_t<std::make_signed_t<typename _Vp::value_type>, _Vp>
bit_width(const _Vp& x) noexcept {
  using _R = rebind_t<std::make_signed_t<typename _Vp::value_type>, _Vp>;
  return _R(detail::cw<std::numeric_limits<typename _Vp::value_type>::digits> - countl_zero(x));
}

/// @brief Compute the smallest integral power of two not smaller than each element.
/// @tparam _Vp The input vector type.
/// @param x The input vector.
/// @return An unsigned vector containing the element-wise bit ceilings.
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

/// @brief Compute the largest integral power of two not greater than each element.
/// Zero input elements produce zero output elements.
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return An unsigned vector containing the element-wise bit floors.
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

/// @} cpp26_bit

/**
 * @defgroup cpp29_p3772 Bit reversal
 * @brief Reverse the bits within each vector element.
 *
 * Proposed in P3772R0.
 *
 * @see https://wg21.link/P3772
 */

/// @brief Reverse the bits within each element.
/// @ingroup cpp29_p3772
/// @tparam _Vp The input vector type.
/// @param x The input vector.
/// @return A vector containing the bit-reversed elements.
template<vec_unsigned_integer _Vp>
constexpr _Vp bit_reverse(const _Vp& x) noexcept {
  if (std::is_constant_evaluated())
    return detail::bitreverse(generic_tag{}, x);
  else
    return detail::bitreverse(target, x); }

/// @brief Reverse the bytes within each integral element.
/// @ingroup cpp26_bit
/// @tparam _Vp The input vector type.
/// @param v The input vector.
/// @return A vector containing the byte-reversed elements.
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

/**
 * @defgroup cpp29_p4010 Funnel shifts
 * @brief Shift concatenated pairs of elements and return one element-width result.
 *
 * Proposed in P4010.
 *
 * @see https://wg21.link/P4010
 */

/// @addtogroup cpp29_p4010
/// @{

/// @brief Funnel shift corresponding element pairs right.
/// Conceptually concatenate each high element with its corresponding low element,
/// shift right, and return the low N bits, where N is the element bit width.
/// @tparam _Up The unsigned vector type.
/// @tparam _Sp The integral shift-count vector type.
/// @param high The upper elements of the concatenation.
/// @param low The lower elements of the concatenation.
/// @param s The element-wise shift counts.
/// @pre For every element i, 0 <= s[i] < N.
/// @return A vector containing the funnel-shift results.
template<vec_unsigned_integer _Up, vec_integral _Sp>
  requires (sizeof(typename _Up::value_type) == sizeof(typename _Sp::value_type))
constexpr _Up funnel_shift_right(const _Up& high, const _Up& low, const _Sp& s)
{
  if (std::is_constant_evaluated())
    return _Up([=](auto i) { return typename _Up::value_type(detail::funnel_shift_right(high[i], low[i], s[i])); });
  else
    return detail::fsr(target, high, low, s);
}

/// @brief Funnel shift every element pair right using a shared shift count.
/// @param high The upper elements of the concatenation.
/// @param low The lower elements of the concatenation.
/// @param s The shared shift count.
/// @pre 0 <= s < N, where N is the element bit width.
/// @return A vector containing the funnel-shift results.
template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up funnel_shift_right(const _Up& high, const _Up& low, _Sp s)
  { return funnel_shift_right(high, low, _Up(typename _Up::value_type(s))); }

/// @brief Funnel shift corresponding element pairs left.
/// Conceptually concatenate each high element with its corresponding low element,
/// shift left, and return the high N bits, where N is the element bit width.
/// @tparam _Up The unsigned vector type.
/// @tparam _Sp The integral shift-count vector type.
/// @param high The upper elements of the concatenation.
/// @param low The lower elements of the concatenation.
/// @param s The element-wise shift counts.
/// @pre For every element i, 0 <= s[i] < N.
/// @return A vector containing the funnel-shift results.
template<vec_unsigned_integer _Up, vec_integral _Sp>
  requires (sizeof(typename _Up::value_type) == sizeof(typename _Sp::value_type))
constexpr _Up funnel_shift_left(const _Up& high, const _Up& low, const _Sp& s)
{
  if (std::is_constant_evaluated())
    return _Up([=](auto i) { return typename _Up::value_type(detail::funnel_shift_left(high[i], low[i], s[i])); });
  else
    return detail::fsl(target, high, low, s);
}

/// @brief Funnel shift every element pair left using a shared shift count.
/// @param high The upper elements of the concatenation.
/// @param low The lower elements of the concatenation.
/// @param s The shared shift count.
/// @pre 0 <= s < N, where N is the element bit width.
/// @return A vector containing the funnel-shift results.
template<vec_unsigned_integer _Up, std::integral _Sp>
constexpr _Up funnel_shift_left(const _Up& high, const _Up& low, _Sp s)
  { return funnel_shift_left(high, low, _Up(typename _Up::value_type(s))); }

/// @} cpp29_p4010

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