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

/// \defgroup simd_utility utility header overloads for simd
/// @brief SIMD versions of functions which appear in C++20 utility header.

/// @brief Converts an enumeration to its underlying type.
/// @tparam _Tp The enumeration type
/// @tparam _Abi The ABI of the simd
/// @param x The simd value of enum type to convert
/// @return A simd value of the underlying type of the enum
template<typename _Tp, typename _Abi>
requires (std::is_enum_v<_Tp>)
constexpr auto to_underlying(const basic_vec<_Tp, _Abi>& x) { return simd_bit_cast<std::underlying_type_t<_Tp>>(x); }

/// @brief Converts a std::byte to an integer type.
/// @tparam _Tp The integral type to convert to
/// @tparam _Vp The simd type to convert from, which must be a simd of std::byte
/// @param b The simd value of std::byte type to convert
/// @return A simd value of the specified integer type
template<std::integral _Tp, vec_of<std::byte> _Vp>
constexpr rebind_t<_Tp, _Vp> to_integer(const _Vp& b) noexcept { return rebind_cast<_Tp>(b); }

} // namespace _XVEC_NAMESPACE::simd
