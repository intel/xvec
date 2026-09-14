//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <immintrin.h>

#include <cstdint> // std::uint8_t

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>
#include <xvec/detail/core.hpp>

namespace _XVEC_NAMESPACE::simd::x86 {

/// Concepts for detecting vectors which fit into specific sizes of registers.
///@{
template<typename _Vec, typename _Tp = typename _Vec::value_type>
concept xmm_register = (sizeof(_Vec) <= 16) && (std::same_as<_Tp, typename _Vec::value_type>);

template<typename _Vec, typename _Tp = typename _Vec::value_type>
concept ymm_register = (sizeof(_Vec) > 16 && sizeof(_Vec) <= 32) && (std::same_as<_Tp, typename _Vec::value_type>);

template<typename _Vec, typename _Tp = typename _Vec::value_type>
concept zmm_register = (sizeof(_Vec) > 32 && sizeof(_Vec) <= 64) && (std::same_as<_Tp, typename _Vec::value_type>);
///@}

#if defined(__GFNI__)
/// Compute an affine transformation in the Galois Field 2^8. An affine
/// transformation is defined by A * x + b, where A represents an 8 by 8 bit
/// matrix, x represents an 8-bit vector, and b is a constant immediate byte.

/// This function call wraps a call directly to the underlying GFNI instruction
/// using 64-bit elements throughout. It allows the underlying GFNI instruction
/// to be used in any way that the hardware supports. It requires 8 bytes to be
/// supplied for every 8x8 matrix element. It allows arbitrary numbers of
/// byte/matrix groups, and will call the intrinsic as many times as needed for
/// all the groups.
template<uint8_t imm_byte = 0, vec_type _Bytes, vec_type _Matrix>
requires ((sizeof(typename _Bytes::value_type) == 1) &&
          (sizeof(typename _Matrix::value_type) == 8) &&
          ((_Matrix::size() * 8) == _Bytes::size()))
inline _Bytes gf2p8affine(const _Bytes& b, const _Matrix& m)
{
  using namespace detail;

  target_overloads impl {
    [=]<xmm_register<std::uint8_t> _Vec>(_Vec x, _Vec y)
      { return _mm_gf2p8affine_epi64_epi8 (x.to_register(), y.to_register(), imm_byte); },
    [=]<ymm_register<std::uint8_t> _Vec>(_Vec x, _Vec y)
      { return _mm256_gf2p8affine_epi64_epi8 (x.to_register(), y.to_register(), imm_byte); },
    [=]<zmm_register<std::uint8_t> _Vec>(_Vec x, _Vec y)
      { return _mm512_gf2p8affine_epi64_epi8 (x.to_register(), y.to_register(), imm_byte); },
 
    [=](auto unhandled, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled; }
  };

  return std::bit_cast<_Bytes>(chunked_invoke([=](auto x, auto y) { return decltype(x)(impl(x, y)); }, 
                                              simd_bit_cast<uint8_t>(b), simd_bit_cast<uint8_t>(m)));
}

/// This variant of the GFNI affine instruction provides an easy way of
/// accessing the most common use case, where every input byte element is
/// multiplied by a single 8x8 bit matrix.
template<uint8_t imm_byte = 0, vec_type _Bytes>
requires (sizeof(typename _Bytes::value_type) == 1) // Allow any type of single byte element to be transformed.
inline _Bytes gf2p8affine(const _Bytes& b, uint64_t m)
{
  constexpr int roundedTo8 = ((_Bytes::size() + 7) / 8) * 8;
  auto resizeTo8 = permute<roundedTo8>(b, detail::perm_uninitResize);
  auto r = gf2p8affine<imm_byte>(resizeTo8, vec<uint64_t, roundedTo8 / 8>(m));
  return take<(int)_Bytes::size()>(r);
}

#else
inline auto gf2p8affine(auto unhandled, auto); // Does nothing for other platforms to allow link-time detection.
#endif

} // End of simd::x86 namespace

namespace _XVEC_NAMESPACE::simd::detail {
  using x86::xmm_register;
  using x86::ymm_register;
  using x86::zmm_register;
}
