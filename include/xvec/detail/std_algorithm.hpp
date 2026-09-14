//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <concepts>
#include <utility> // std::pair, std::make_pair

namespace _XVEC_NAMESPACE::simd
{

/// @brief Compute the smaller of each respective vec element from the two input arguments.
/// @ingroup simd_alg
/// @tparam _Vp The vec type
/// @param lhs The left-hand value to compare
/// @param rhs The right-hand value to compare
/// @return A new vec where each element is the smaller of the two respective elements from the input values.
template<vec_type _Vp>
requires std::totally_ordered<typename _Vp::value_type>
constexpr _Vp min(const _Vp& lhs, const _Vp& rhs) noexcept {
  if (std::is_constant_evaluated() || !detail::builtin_vectorizable<typename _Vp::value_type>)
    return _Vp([=](auto i) { using std::min; return min(lhs[i], rhs[i]); });
  else
    return detail::minimum(target, lhs, rhs);
}

/// @brief Compute the larger of each respective vec element from the two input arguments.
/// @ingroup simd_alg
/// @tparam _Vp The vec type
/// @param lhs The left-hand value to compare
/// @param rhs The right-hand value to compare
/// @return A new vec where each element is the larger of the two respective elements from the input values.
template<vec_type _Vp>
requires std::totally_ordered<typename _Vp::value_type>
constexpr _Vp max(const _Vp& lhs, const _Vp& rhs) noexcept {
  if (std::is_constant_evaluated() || !detail::builtin_vectorizable<typename _Vp::value_type>)
    return _Vp([=](auto i) { using std::max; return max(lhs[i], rhs[i]); });
  else
    return detail::maximum(target, lhs, rhs);
}

/// @brief Compute a pair of vec values giving the smaller and larger
/// respective of each corresponding vec element from the two input arguments.
/// @ingroup simd_alg
/// @tparam _Vp The vec type.
/// @param lhs The left-hand value to compare
/// @param rhs The right-hand value to compare
/// @return A pair of vec values representing the results of the min and max functions applied to the inputs.
template<vec_totally_ordered _Vp>
requires std::totally_ordered<typename _Vp::value_type>
constexpr std::pair<_Vp, _Vp>
minmax(const _Vp& lhs, const _Vp& rhs) noexcept { return std::make_pair(min(lhs, rhs), max(lhs, rhs)); }

/// \brief Clamp the input values to be contained within the range [low,high]. If a
/// value is less than low it will return low. If a value is greater than high
/// it will return high. Other values are unmodified.
/// \ingroup simd_alg
/// \tparam _Vp The vec type.
/// \param values The values to clamp.
/// \param low The low value in the clamping range.
/// \param high The high value in the clamping range.
/// \return A vec where each element is clamped to the range [low, high].
template<vec_totally_ordered _Vp>
requires std::totally_ordered<typename _Vp::value_type>
constexpr _Vp
clamp(const _Vp& values, const _Vp& low, const _Vp& high) { return min(max(values, low), high); }

/// @brief Given two vec values, choose one or the other element as output for each element
/// position. select works like (is_bit_set ? a : b) for each element.
/// When the corresponding bit is set in the mask the output element is chosen
/// to be from a, otherwise from b. The second operand defaults to zero if no
/// value is supplied.
/// @ingroup simd_select
/// @tparam _M The mask type
/// @tparam _T The type of the first value
/// @tparam _U The type of the second value
/// @param m The mask indicating which elements to select
/// @param if_true The value to use if the mask bit is set
/// @param if_false The value to use if the mask bit is clear.
/// @return A vec where each element is selected from if_true or if_false according to the mask.
template<mask_type _M, class _T, class _U>
constexpr auto select(const _M& m, const _T& if_true, const _U& if_false) noexcept -> decltype(simd_select_impl(m, if_true, if_false))
  { return simd_select_impl(m, if_true, if_false); }

/// @brief Select between two values based on a boolean condition (scalar version).
/// @tparam _Tp The type of the first value
/// @tparam _Up The type of the second value
/// @param c The condition
/// @param a The value to use if c is true
/// @param b The value to use if c is false
/// @return a if c is true, otherwise b
template<std::copyable _Tp, std::copyable _Up>
constexpr auto select(bool c, const _Tp& a, const _Up& b) -> std::remove_cvref_t<decltype(c ? a : b)> { return c ? a : b; }

} // namespace _XVEC_NAMESPACE::simd
