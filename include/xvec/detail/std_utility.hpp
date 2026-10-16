//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <utility>
#include <cstddef>

namespace _XVEC_NAMESPACE::simd
{

/**
 * @defgroup cpp29_p2964 SIMD support for enumerations and std::byte
 * @brief Facilities associated with the additional element types proposed in P2964.
 *
 * @see https://wg21.link/P2964
 * @{
 */

/// @brief Convert each enumeration element to its underlying type.
/// @tparam _Tp The enumeration type.
/// @tparam _Abi The ABI of the vector.
/// @param x The vector of enumeration values to convert.
/// @return A vector whose elements have the enumeration's underlying type.
template<typename _Tp, typename _Abi>
requires (std::is_enum_v<_Tp>)
constexpr auto to_underlying(const basic_vec<_Tp, _Abi>& x) { return simd_bit_cast<std::underlying_type_t<_Tp>>(x); }

/// @brief Convert each std::byte element to an integer type.
/// @tparam _Tp The integral type to convert to.
/// @tparam _Vp The input vector type, whose elements must be std::byte.
/// @param b The vector of byte values to convert.
/// @return A vector whose elements have the specified integer type.
template<std::integral _Tp, vec_of<std::byte> _Vp>
constexpr rebind_t<_Tp, _Vp> to_integer(const _Vp& b) noexcept { return rebind_cast<_Tp>(b); }

/// @} cpp29_p2964

} // namespace _XVEC_NAMESPACE::simd