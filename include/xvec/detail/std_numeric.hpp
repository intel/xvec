//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <concepts>
#include <type_traits>

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>

namespace _XVEC_NAMESPACE::simd
{
/**
 * @defgroup cpp29_p2956 SIMD saturating arithmetic
 * @brief Saturating arithmetic and element conversion for integral vectors.
 *
 * Proposed in P2956, "Add saturating library support to std::simd".
 *
 * @see https://wg21.link/P2956
 * @{
 */

/// @brief Compute the saturated addition of each respective pair of SIMD elements.
/// @tparam _Vp The integral basic_vec type.
/// @param lhs The first basic_vec value.
/// @param rhs The second basic_vec value.
/// @return A basic_vec containing the saturated element-wise sums.
template<vec_integral _Vp> constexpr _Vp saturating_add(const _Vp& lhs, const _Vp& rhs) noexcept {
  if (std::is_constant_evaluated())
    return _Vp([&](auto i) { return detail::saturating_add(lhs[i], rhs[i]); });
  else
    return detail::saturating_add(target, lhs, rhs);
}

/// @brief Compute the saturated subtraction of each respective pair of SIMD elements.
/// @tparam _Vp The integral basic_vec type.
/// @param lhs The first basic_vec value.
/// @param rhs The second basic_vec value.
/// @return A basic_vec containing the saturated element-wise differences.
template<vec_integral _Vp> constexpr _Vp saturating_sub(const _Vp& lhs, const _Vp& rhs) noexcept {
  if (std::is_constant_evaluated())
    return _Vp([&](auto i) { return detail::saturating_sub(lhs[i], rhs[i]); });
  else
    return detail::saturating_sub(target, lhs, rhs);
}

/// @brief Compute the saturated multiplication of each respective pair of SIMD elements.
/// @tparam _Vp The integral basic_vec type.
/// @param lhs The first basic_vec value.
/// @param rhs The second basic_vec value.
/// @return A basic_vec containing the saturated element-wise products.
template<vec_integral _Vp> constexpr _Vp saturating_mul(const _Vp& lhs, const _Vp& rhs) noexcept
  { return _Vp([&](auto i) { return detail::saturating_mul(lhs[i], rhs[i]); }); }

/// @brief Compute the saturated division of each respective pair of SIMD elements.
/// @tparam _Vp The integral basic_vec type.
/// @param lhs The dividend basic_vec value.
/// @param rhs The divisor basic_vec value.
/// @return A basic_vec containing the saturated element-wise quotients.
template<vec_integral _Vp> constexpr _Vp saturating_div(const _Vp& lhs, const _Vp& rhs) noexcept
  { return _Vp([&](auto i) { return detail::saturating_div(lhs[i], rhs[i]); }); }

/// @brief Compute a basic_vec where every element is converted to the new type, or
/// if the value can't be represented then the largest or smallest value of that
/// type is used instead, whichever is closer to the original value.
/// @tparam _Up The target integral element type.
/// @tparam _Vp The integral basic_vec type.
/// @param v The basic_vec value to convert.
/// @return A basic_vec value in which each element is the saturating cast of the respective input value.
template<std::integral _Up, vec_integral _Vp>
constexpr rebind_t<_Up, _Vp> saturating_cast(const _Vp& v) noexcept {
  if (std::is_constant_evaluated())
    return rebind_t<_Up, _Vp>([&](auto i) { return detail::saturating_cast<_Up>(v[i]); });
  else
    return detail::saturating_cast<_Up>(target, v);
}

/// @} cpp29_p2956

} // namespace _XVEC_NAMESPACE::simd
