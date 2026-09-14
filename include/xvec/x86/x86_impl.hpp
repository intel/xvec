//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <immintrin.h>
#include <bitset>
#include <cstdint>
#include <cstddef> // std::size_t
#include <concepts> // std::floating_point, std::unsigned_integral
#include <complex>
#include <type_traits>
#include <algorithm> // std::min
#include <functional> // std::equal_to, std::less, etc.
#include <bit> // std::popcount
#include <limits> // std::numeric_limits
#include <memory> // std::to_address
#include <utility> // std::pair

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>
#include <xvec/detail/core.hpp>

#include <xvec/x86/permute.hpp>

namespace _XVEC_NAMESPACE::simd {

namespace detail
{

#if defined(__AVX512F__)
template<typename _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi>
select_if_else(x86_avx512_tag, const typename basic_vec<_Tp, _Abi>::mask_type& if_mask,
               const basic_vec<_Tp, _Abi>& true_value, const basic_vec<_Tp, _Abi>& false_value)
{
  target_overloads impl {
    [=]<xmm_register<std::uint8_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm_mask_blend_epi8(__mmask16(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<ymm_register<std::uint8_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm256_mask_blend_epi8(__mmask32(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<zmm_register<std::uint8_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm512_mask_blend_epi8(__mmask64(m.to_register()), if_false.to_register(), if_true.to_register()); },

    [=]<xmm_register<std::uint16_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm_mask_blend_epi16(__mmask8(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<ymm_register<std::uint16_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm256_mask_blend_epi16(__mmask16(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<zmm_register<std::uint16_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm512_mask_blend_epi16(__mmask32(m.to_register()), if_false.to_register(), if_true.to_register()); },

    [=]<xmm_register<std::uint32_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm_mask_blend_epi32(__mmask8(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<ymm_register<std::uint32_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm256_mask_blend_epi32(__mmask8(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<zmm_register<std::uint32_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm512_mask_blend_epi32(__mmask16(m.to_register()), if_false.to_register(), if_true.to_register()); },

    [=]<xmm_register<std::uint64_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm_mask_blend_epi64(__mmask8(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<ymm_register<std::uint64_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm256_mask_blend_epi64(__mmask8(m.to_register()), if_false.to_register(), if_true.to_register()); },
    [=]<zmm_register<std::uint64_t> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm512_mask_blend_epi64(__mmask8(m.to_register()), if_false.to_register(), if_true.to_register()); },

    [=]<xmm_register<unsigned __int128> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return (m.to_register() & 1) == 0 ? if_false.to_register() : if_true.to_register(); },
    [=]<ymm_register<unsigned __int128> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm256_mask_blend_epi64(__mmask8(dupBits(m.to_register())), if_false.to_register(), if_true.to_register()); },
    [=]<zmm_register<unsigned __int128> _Vec>(_Vec if_true, _Vec if_false, auto m)
      { return _mm512_mask_blend_epi64(__mmask8(dupBits(m.to_register())), if_false.to_register(), if_true.to_register()); },

    [=](auto unhandled, auto, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled; }
  };

  auto wrapper = [=]<typename _Vec>(_Vec ift, _Vec iff, auto ifm) {
    auto r = _Vec(impl(ift, iff, ifm));
    return simd_bit_cast<_Tp>(r);
  };

  constexpr int _Np = basic_vec<_Tp, _Abi>::size;
  using _C = container_for_type<_Tp>;

  auto trueAsC = vec_as_container(true_value);
  auto falseAsC = vec_as_container(false_value);

  if (std::is_constant_evaluated())
  {
    // operator? doesn't work in constexpr for Clang so implement in terms of
    // bitwise operations instead with a wide mask (all 0/1 bits). Note that
    // `-mask` is defined in terms of select, so the mask must be built
    // explicitly here.
    auto wideM = vec<_C, _Np>([=](auto i) -> _C { if (if_mask[i]) return ~_C(); else return _C(); }).to_builtin();
    basic_vec<_C, _Abi> r = (wideM & trueAsC.to_builtin()) | ((~wideM) & falseAsC.to_builtin());
    return simd_bit_cast<_Tp>(r);
  }
  else
    return chunked_invoke(wrapper, trueAsC, falseAsC, mask<_C, _Np>(if_mask));
}
#endif

inline target_overloads msb_to_bitmask_impl {
  // Intel SSE
  [](xmm_register<std::uint8_t>  auto m) -> std::uint16_t { return _mm_movemask_epi8(m.to_register()); },
  [](xmm_register<std::uint16_t> auto m) -> std::uint8_t {
    // No 16-bit equivalent of movemask in Intel SSE so put all the upper bits of the mask
    // elements into the lower 64-bits and grab those. All the other elements
    // are zeroed.
    const auto B = simd_bit_cast<std::uint8_t>(m);
    const auto reqBits = permute<16>(B, [](auto idx) { return idx < decltype(m)::size() ? idx * 2 : zero_element; });
    return _mm_movemask_epi8(reqBits.to_register());
  },
  [](xmm_register<std::uint32_t> auto m) -> std::uint8_t { return _mm_movemask_ps(__m128(m.to_register()));},
  [](xmm_register<std::uint64_t> auto m) -> std::uint8_t { return _mm_movemask_pd(__m128d(m.to_register()));},
  [](xmm_register<unsigned __int128> auto m) -> std::uint8_t { return _mm_movemask_pd(__m128d(m.to_register())) >> 1;},

#if defined(__AVX2__)
  // Intel AVX2
  [](ymm_register<std::uint8_t> auto m) -> std::uint32_t { return _mm256_movemask_epi8(m.to_register()); },
  [](ymm_register<std::uint16_t> auto m) -> std::uint16_t { return _pext_u64(_mm256_movemask_epi8(m.to_register()), 0xAAAAAAAA); },
#elif defined(__AVX__)
  // Intel AVX - doesn't have 16-bit movemask, or 8-bit beyond the first 128-bit lane.
  [](ymm_register<std::uint8_t> auto m) -> std::uint32_t {
    // Process upper and lower halves separately and then combine them.
    auto lower = _mm_movemask_epi8(_mm256_castsi256_si128(m.to_register()));
    auto upper = _mm_movemask_epi8(_mm256_extractf128_si256(m.to_register(), 1));
    return std::uint32_t(lower) | (std::uint32_t(upper) << 16);
  },
  [](ymm_register<std::uint16_t> auto m) -> std::uint16_t {
    // Convert each the top byte of each 16-bit value into an 8-bit value and extract their sign bits.
    auto to8 = rebind_cast<std::uint8_t>(m >> 8);
    return _mm_movemask_epi8(to8.to_register());
  },
#endif

#if defined(__AVX__)
  [](ymm_register<std::uint32_t> auto m) -> std::uint8_t { return _mm256_movemask_ps(__m256(m.to_register()));},
  [](ymm_register<std::uint64_t> auto m) -> std::uint8_t { return _mm256_movemask_pd(__m256d(m.to_register()));},
  [](ymm_register<unsigned __int128> auto m) -> std::uint8_t {
    auto fourMsbs = _mm256_movemask_pd(__m256d(m.to_register()));
    // Only need bits 1 and 3.
    #if defined(__BMI2__)
      return _pext_u64(fourMsbs, 0b1010);
    #else
      auto masked4 = (fourMsbs & 0b1010) >> 1;
      return ((masked4 >> 1) | masked4) & 0b11;
    #endif
  },
#endif

  [](auto unhandled) -> unsigned long { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return 0;}
};

template<mask_type _Mp>
constexpr std::bitset<_Mp::size>
mask_to_bitset(x86_sse_tag, const _Mp& mask) noexcept
{
  constexpr int _Np = _Mp::size;
  using _R = std::bitset<_Np>;

  // Given a subset of a vec_impl, extract the raw bits and insert into the correct result position.
  _R result = {};
  auto processBlock = [&](auto v, simd_size_type idx) {
    auto msbs = msb_to_bitmask_impl(vec_as_container(v)) & ((1ULL << v.size()) - 1);
    result |= _R(msbs) << idx;
  };
  chunked_invoke(processBlock, mask);
  return result;
}

template<mask_type _Mp>
constexpr std::bitset<_Mp::size>
mask_to_bitset(x86_avx512_tag, const _Mp& mask) noexcept
{
  constexpr int _Np = _Mp::size;
  std::bitset<_Np> result;

  // Build the bitset up 64-bits at a time. Each 64-bit group is extracted from the extended int and
  // then inserted into the bitset.
#pragma unroll
  for (simd_size_type i = 0; i<((_Np + 63) / 64); ++i)
  {
    const auto b = std::bitset<_Np>(std::uint64_t(mask.to_builtin() >> (64 * i)));
    result |= (b << (64 * i));
  }

  return result;
}


template<std::floating_point _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> fma(x86_avx2_tag, const basic_vec<_Tp, _Abi>& x, const basic_vec<_Tp, _Abi>& y, const basic_vec<_Tp, _Abi>& acc)
{
  target_overloads impl {
    // 128-bit (not Intel SSE - only Intel AVX2 onwards has FMA)
    [=](xmm_register<float> auto x, auto y, auto acc) { return _mm_fmadd_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](xmm_register<double> auto x, auto y, auto acc) { return _mm_fmadd_pd(x.to_register(), y.to_register(), acc.to_register()); },

    // 256-bit
    [=](ymm_register<float> auto x, auto y, auto acc) { return _mm256_fmadd_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](ymm_register<double> auto x, auto y, auto acc) { return _mm256_fmadd_pd(x.to_register(), y.to_register(), acc.to_register()); },

    // 512-bit
#if defined(__AVX512F__)
    [=](zmm_register<float> auto x, auto y, auto acc) { return _mm512_fmadd_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](zmm_register<double> auto x, auto y, auto acc) { return _mm512_fmadd_pd(x.to_register(), y.to_register(), acc.to_register()); },
#endif

    [=](auto unhandled, auto, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b, auto c) { return decltype(a)(impl(a, b, c)); }; // Return a vec from an intrinsic.
  return chunked_invoke(wrapper, x, y, acc);
}

template<std::floating_point _Tp, typename _Abi>
constexpr
basic_vec<_Tp, _Abi>
fmaddsub(x86_avx2_tag, const basic_vec<_Tp, _Abi>& x, const basic_vec<_Tp, _Abi>& y, const basic_vec<_Tp, _Abi>& acc)
{
  target_overloads impl {
    // 128-bit (not Intel SSE - only Intel AVX2 onwards has FMA)
    [=](xmm_register<float> auto x, auto y, auto acc) { return _mm_fmaddsub_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](xmm_register<double> auto x, auto y, auto acc) { return _mm_fmaddsub_pd(x.to_register(), y.to_register(), acc.to_register()); },

    // 256-bit
    [=](ymm_register<float> auto x, auto y, auto acc) { return _mm256_fmaddsub_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](ymm_register<double> auto x, auto y, auto acc) { return _mm256_fmaddsub_pd(x.to_register(), y.to_register(), acc.to_register()); },

    // 512-bit
#if defined(__AVX512F__)
    [=](zmm_register<float> auto x, auto y, auto acc) { return _mm512_fmaddsub_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](zmm_register<double> auto x, auto y, auto acc) { return _mm512_fmaddsub_pd(x.to_register(), y.to_register(), acc.to_register()); },
#endif

    [=](auto unhandled, auto, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b, auto c) { return decltype(a)(impl(a, b, c)); }; // Return a vec from an intrinsic.
  return chunked_invoke(wrapper, x, y, acc);
}

template<std::floating_point _Tp, typename _Abi>
constexpr
basic_vec<_Tp, _Abi>
fmsubadd(x86_avx2_tag, const basic_vec<_Tp, _Abi>& x, const basic_vec<_Tp, _Abi>& y, const basic_vec<_Tp, _Abi>& acc)
{
  target_overloads impl {
    // 128-bit (not Intel SSE - only Intel AVX2 onwards has FMA)
    [=](xmm_register<float> auto x, auto y, auto acc) { return _mm_fmsubadd_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](xmm_register<double> auto x, auto y, auto acc) { return _mm_fmsubadd_pd(x.to_register(), y.to_register(), acc.to_register()); },

    // 256-bit
    [=](ymm_register<float> auto x, auto y, auto acc) { return _mm256_fmsubadd_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](ymm_register<double> auto x, auto y, auto acc) { return _mm256_fmsubadd_pd(x.to_register(), y.to_register(), acc.to_register()); },

    // 512-bit
#if defined(__AVX512F__)
    [=](zmm_register<float> auto x, auto y, auto acc) { return _mm512_fmsubadd_ps(x.to_register(), y.to_register(), acc.to_register()); },
    [=](zmm_register<double> auto x, auto y, auto acc) { return _mm512_fmsubadd_pd(x.to_register(), y.to_register(), acc.to_register()); },
#endif

    [=](auto unhandled, auto, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b, auto c) { return decltype(a)(impl(a, b, c)); }; // Return a vec from an intrinsic.
  return chunked_invoke(wrapper, x, y, acc);
}

#if defined(__AVX512FP16__)

template<vec_of<_Float16> _Vp>
constexpr _Vp fma(x86_avxspr_tag, const _Vp& x, const _Vp& y, const _Vp& acc) {
  target_overloads impl {
    [=](xmm_register<_Float16> auto x, auto y, auto acc) { return _mm_fmadd_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](ymm_register<_Float16> auto x, auto y, auto acc) { return _mm256_fmadd_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](zmm_register<_Float16> auto x, auto y, auto acc) { return _mm512_fmadd_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](auto unhandled, auto, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b, auto c) { return decltype(a)(impl(a, b, c)); }; // Return a vec from an intrinsic.
  return chunked_invoke(wrapper, x, y, acc);
}

template<vec_of<std::complex<_Float16>> _Vp>
constexpr _Vp fma(x86_avxspr_tag, const _Vp& x, const _Vp& y, const _Vp& acc) {
  target_overloads impl {
    [=](xmm_register<_Float16> auto x, auto y, auto acc) { return _mm_fmadd_pch(x.to_register(), y.to_register(), acc.to_register()); },
    [=](ymm_register<_Float16> auto x, auto y, auto acc) { return _mm256_fmadd_pch(x.to_register(), y.to_register(), acc.to_register()); },
    [=](zmm_register<_Float16> auto x, auto y, auto acc) { return _mm512_fmadd_pch(x.to_register(), y.to_register(), acc.to_register()); },
    [=](auto unhandled, auto, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b, auto c) { return decltype(a)(impl(a, b, c)); }; // Return a vec from an intrinsic.
  auto t =  chunked_invoke(wrapper, simd_bit_cast<_Float16>(x), simd_bit_cast<_Float16>(y), simd_bit_cast<_Float16>(acc));
  return simd_bit_cast<std::complex<_Float16>>(t);
}

template<vec_of<_Float16> _Vp>
constexpr _Vp fmaddsub(x86_avxspr_tag, const _Vp& x, const _Vp& y, const _Vp& acc) {
  target_overloads impl {
    [=](xmm_register<_Float16> auto x, auto y, auto acc) { return _mm_fmaddsub_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](ymm_register<_Float16> auto x, auto y, auto acc) { return _mm256_fmaddsub_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](zmm_register<_Float16> auto x, auto y, auto acc) { return _mm512_fmaddsub_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](auto unhandled, auto, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b, auto c) { return decltype(a)(impl(a, b, c)); }; // Return a vec from an intrinsic.
  return chunked_invoke(wrapper, x, y, acc);
}

template<vec_of<_Float16> _Vp>
constexpr _Vp
fmsubadd(x86_avxspr_tag, const _Vp& x, const _Vp& y, const _Vp& acc)
{
  target_overloads impl {
    [=](xmm_register<_Float16> auto x, auto y, auto acc) { return _mm_fmsubadd_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](ymm_register<_Float16> auto x, auto y, auto acc) { return _mm256_fmsubadd_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](zmm_register<_Float16> auto x, auto y, auto acc) { return _mm512_fmsubadd_ph(x.to_register(), y.to_register(), acc.to_register()); },
    [=](auto unhandled, auto, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b, auto c) { return decltype(a)(impl(a, b, c)); }; // Return a vec from an intrinsic.
  return chunked_invoke(wrapper, x, y, acc);
}

template<vec_of<std::complex<_Float16>> _Vp>
constexpr _Vp builtin_operator(x86_avxspr_tag, const _Vp& x, const _Vp& y, std::multiplies<>)
{
  target_overloads impl {
    [=](xmm_register<_Float16> auto x, auto y) { return _mm_fmul_pch(x.to_register(), y.to_register()); },
    [=](ymm_register<_Float16> auto x, auto y) { return _mm256_fmul_pch(x.to_register(), y.to_register()); },
    [=](zmm_register<_Float16> auto x, auto y) { return _mm512_fmul_pch(x.to_register(), y.to_register()); },
    [=](auto unhandled, auto) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  auto wrapper = [=](auto a, auto b) { return decltype(a)(impl(a, b)); }; // Return a vec from an intrinsic.
  auto t =  chunked_invoke(wrapper, simd_bit_cast<_Float16>(x), simd_bit_cast<_Float16>(y));
  return simd_bit_cast<std::complex<_Float16>>(t);
}

#endif

///@{
/// \brief Rounding operations (ceil, floor, etc.)

template<RoundOp _Op, typename _Tp, typename _Abi>
requires (detail::is_floating_point_v<_Tp>)
constexpr basic_vec<_Tp, _Abi> round_op(x86_tag, const basic_vec<_Tp, _Abi>& v)
{
  [[maybe_unused]] constexpr auto sseOp = std::array<int, 5>{
    _MM_FROUND_TO_POS_INF, _MM_FROUND_TO_NEG_INF, _MM_FROUND_TO_ZERO,
    _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_CUR_DIRECTION
  }[int(_Op)];
  [[maybe_unused]] constexpr auto avx512Op = std::array<int, 5>{
    _MM_FROUND_CEIL, _MM_FROUND_FLOOR, _MM_FROUND_TO_ZERO,
    _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_CUR_DIRECTION
  }[int(_Op)];

  target_overloads impl {
    // Intel SSE
    [=](xmm_register<float> auto x)  { return _mm_round_ps(x.to_register(), sseOp); },
    [=](xmm_register<double> auto x) { return _mm_round_pd(x.to_register(), sseOp); },

    // Intel AVX
    [=](ymm_register<float> auto x)  { return _mm256_round_ps(x.to_register(), sseOp); },
    [=](ymm_register<double> auto x) { return _mm256_round_pd(x.to_register(), sseOp); },

    // Intel AVX-512
#if defined(__AVX512F__)
    [=](zmm_register<float> auto x)  { return _mm512_roundscale_ps(x.to_register(), sseOp); },
    [=](zmm_register<double> auto x) { return _mm512_roundscale_pd(x.to_register(), sseOp); },
#endif

#if defined(__AVX512FP16__)
    [=](xmm_register<_Float16> auto x) { return _mm_roundscale_ph(x.to_register(), avx512Op); },
    [=](ymm_register<_Float16> auto x) { return _mm256_roundscale_ph(x.to_register(), avx512Op); },
    [=](zmm_register<_Float16> auto x) { return _mm512_roundscale_ph(x.to_register(), avx512Op); },
#endif

    [=](auto unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled; }
  };

  if constexpr (_Op == detail::RoundOp::ROUND)
  {
    // Unfortunately the round/roundscale intrinsics don't have a mode which
    // matches C++ round function, so it needs to be built manually.
    constexpr _Tp almost_half = [] {
      if constexpr (sizeof(_Tp) == 2) return static_cast<_Tp>(0x1.ffcp-2);
      else if constexpr (sizeof(_Tp) == 4) return 0x1.fffffep-2f;
      else return 0x1.fffffffffffffp-2;
    }();

    return trunc(v + copysign(basic_vec<_Tp, _Abi>(almost_half), v));
  }
  else if constexpr (is_fp16_v<_Tp> && !hasAvxFp16)
  {
    // When FP16 rounding is needed in emulation, do the rounding in FP32
    // instead. All FP16 values can be converted to a valid FP32 which will round in the same way.
    const auto rounded32 = round_op<_Op>(x86_tag{}, rebind_cast<float>(v));
    return rebind_cast<_Float16>(rounded32);
  }
  else
  {
    auto wrapper = [=](auto x) { return decltype(x)(impl(x)); }; // Return a vec from an intrinsic.
    return chunked_invoke(wrapper, v);
  }
}
///@}

///@{

/// @brief Implement masked stores for Intel SSE and Intel AVX targets.
/// @tparam _Vp the type of the source.
/// @tparam _Up The element type for the vec being stored.
/// @tparam _Extent The extent of the output span.
/// @tparam _Flags Memory flags. This can be used to enable or disable boundary checking.
/// @param value_original The vec value to write to memory.
/// @param to The output buffer as a span.
/// @param mask_original The mask value controlling which vec elements will be written to memory.
/// @param flags Memory flags. This can be used to enable or disable boundary checking.
template<vec_type _Vp, typename _Up, std::size_t _Extent, typename _Flags>
constexpr void store_masked(x86_tag, const _Vp& value_original, std::span<_Up, _Extent> to,
                            const typename _Vp::mask_type& mask_original, _Flags flags)
{
  mandates_for_store(value_original, to, flags);

  using _ToVec = rebind_t<_Up, _Vp>;
  auto from = _ToVec(value_original);
  auto from_mask = typename _ToVec::mask_type(mask_original);

  // This function uses the Intel SSE instruction to write only selected bytes
  // to memory. Bigger registers must be broken down into Intel SSE-sized
  // pieces.
  constexpr auto numSseElements = 16 / sizeof(_Up);
  auto impl = [=]<typename _Vec>(_Vec x, auto m, auto idx) {
    // Note that the mask register is zero extended to ensure no extra mask elements appear in it during `to_register'.
    _mm_maskmoveu_si128(__m128i(x.to_register()), __m128i(grow<numSseElements>(m).to_register()), (char*)(to.data() + idx));
  };

  // Use the mask to limit the bounds of the write if requested by the user.
  auto boundedMask = from_mask;
  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds<decltype(from)>("store", to, from_mask);
  else
    boundedMask &= from_mask.__mask_from_count(to.size());

  chunked_invoke<numSseElements>(impl, vec_as_container(from), boundedMask);
}

/// Special case code for handling 4 and 8 byte masked stores on Intel AVX2. This is faster than using the Intel SSE variant above.
#if defined(__AVX2__)
template<vec_type _Vp, typename _Up, std::size_t _Extent, typename _Flags>
requires (sizeof(_Up) == 4 || sizeof(_Up) == 8)
constexpr void store_masked(x86_avx2_tag, const _Vp& value_original, std::span<_Up, _Extent> to,
                            const typename _Vp::mask_type& mask_original, _Flags flags)
{
  mandates_for_store(value_original, to, flags);

  using _ToVec = rebind_t<_Up, _Vp>;
  auto from = _ToVec(value_original);
  auto from_mask = typename _ToVec::mask_type(mask_original);

  auto impl = [=]<typename _Vec>(_Vec s, auto m, auto idx) {
    // Resize the mask with zeroes to allow partial registers to be written
    // back. Can't use initialised resize because that could introduce unwanted
    // mask elements.
    constexpr auto numAvxElements = 32 / sizeof(_Up);

    const auto rm = grow<numAvxElements>(m);

    // Resize the values to a whole Intel AVX2 register because a partial register can't be allowed into an Intel SSE.
    const auto rc = permute<numAvxElements>(s, perm_uninitResize);

    if constexpr (sizeof(_Up) == 4)
      _mm256_maskstore_epi32(reinterpret_cast<int*>(to.data() + idx), rm.to_register(), __m256i(rc.to_register()));
    else if constexpr (sizeof(_Up) == 8)
      _mm256_maskstore_epi64(reinterpret_cast<long long*>(to.data() + idx), rm.to_register(), __m256i(rc.to_register()));
    else
      static_assert(dependent_false<_Vec>, "Unimplemented masked store");
  };

  // Use the mask to limit the bounds of the write if requested by the user.
  auto boundedMask = from_mask;
  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds<_ToVec>("store", to, from_mask);
  else
    boundedMask &= from_mask.__mask_from_count(to.size());

  chunked_invoke(impl, vec_as_container(from), boundedMask);
}
#endif // __AVX2__

#if defined(__AVX512F__)
/// Store a masked value up to a native register size
template<vec_type _Vp, typename _Up, std::size_t _Extent, typename _Flags>
constexpr void store_masked(x86_avx512_tag, const _Vp& value_original, std::span<_Up, _Extent> to,
                            const typename _Vp::mask_type& mask_original, _Flags flags)
{
  mandates_for_store(value_original, to, flags);

  target_overloads call_insn {
    [=](xmm_register<uint8_t> auto v, auto m, auto p)  { _mm_mask_storeu_epi8(p, m, __m128i(v.to_register())); },
    [=](xmm_register<uint16_t> auto v, auto m, auto p) { _mm_mask_storeu_epi16(p, m, __m128i(v.to_register())); },
    [=](xmm_register<uint32_t> auto v, auto m, auto p) { _mm_mask_storeu_epi32(p, m, __m128i(v.to_register())); },
    [=](xmm_register<uint64_t> auto v, auto m, auto p) { _mm_mask_storeu_epi64(p, m, __m128i(v.to_register())); },

    [=](ymm_register<uint8_t> auto v, auto m, auto p)  { _mm256_mask_storeu_epi8(p, m, __m256i(v.to_register())); },
    [=](ymm_register<uint16_t> auto v, auto m, auto p) { _mm256_mask_storeu_epi16(p, m, __m256i(v.to_register())); },
    [=](ymm_register<uint32_t> auto v, auto m, auto p) { _mm256_mask_storeu_epi32(p, m, __m256i(v.to_register())); },
    [=](ymm_register<uint64_t> auto v, auto m, auto p) { _mm256_mask_storeu_epi64(p, m, __m256i(v.to_register())); },

    [=](zmm_register<uint8_t> auto v, auto m, auto p)  { _mm512_mask_storeu_epi8(p, m, __m512i(v.to_register())); },
    [=](zmm_register<uint16_t> auto v, auto m, auto p) { _mm512_mask_storeu_epi16(p, m, __m512i(v.to_register())); },
    [=](zmm_register<uint32_t> auto v, auto m, auto p) { _mm512_mask_storeu_epi32(p, m, __m512i(v.to_register())); },
    [=](zmm_register<uint64_t> auto v, auto m, auto p) { _mm512_mask_storeu_epi64(p, m, __m512i(v.to_register())); },

    [=](xmm_register<unsigned __int128> auto v, auto m, auto p) { _mm_mask_storeu_epi64(p, dupBits(m), __m128i(v.to_register())); },
    [=](ymm_register<unsigned __int128> auto v, auto m, auto p) { _mm256_mask_storeu_epi64(p, dupBits(m), __m256i(v.to_register())); },
    [=](zmm_register<unsigned __int128> auto v, auto m, auto p) { _mm512_mask_storeu_epi64(p, dupBits(m), __m512i(v.to_register())); },

    [=](auto unhandled, auto, auto) { static_assert(dependent_false<decltype(unhandled)>, "Unimplemented partial masked store"); }
  };

  using _ToVec = rebind_t<_Up, _Vp>;
  auto from = _ToVec(value_original);
  auto from_mask = typename _ToVec::mask_type(mask_original);

  auto impl = [=]<typename _Vec>(_Vec x, auto m, auto idx) {
    // Create a mask which both truncates to the correct number of values and also applies the incoming mask.
    constexpr auto partialMask = typename _Vec::mask_type(true);
    const auto tm = m.to_builtin() & partialMask.to_builtin();
    call_insn(x, tm, to.data() + idx);
  };

  // Use the mask to limit the bounds of the write if requested by the user.
  auto boundedMask = from_mask;
  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds<_ToVec>("store", to, from_mask);
  else
    boundedMask &= from_mask.__mask_from_count(to.size());

  chunked_invoke(impl, vec_as_container(from), boundedMask);
}
#endif    

/// Store up to a native-sized value to memory. All x86 ISAs are handled here.
template<vec_type _Vp, typename _Up, std::size_t _Extent, typename _Flags>
constexpr void store(x86_tag, const _Vp& value_original, std::span<_Up, _Extent> to, _Flags flags)
{
  mandates_for_store(value_original, to, flags);

  auto from = rebind_cast<_Up>(value_original);
  auto ptr = to.data();

  if constexpr (!contains_flag<flag_unchecked>(flags))
  {
    // Handle bounded stores using a masked store which already implements the
    // bounded store efficiently.
    partial_store(from, ptr, to.size(), typename _Vp::mask_type(true), flags);
  }
  else
  {
    checkStaticMemoryBounds<decltype(from)>("store", to);

    // Break the store down into individual pieces which are as large as
    // possible, and which can be written back in individual operations.
    auto impl = [=]<typename _Vec>(_Vec s, auto idx)
    {
      constexpr auto numBytes = _Vec::size * sizeof(_Up);

      // Try a whole register store to begin with.
      if constexpr (numBytes == 16)
        _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + idx), __m128i(s.to_register()));
#if defined(__AVX__)
      else if constexpr (numBytes == 32)
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + idx), __m256i(s.to_register()));
#endif
#if defined(__AVX512F__)
      else if constexpr (numBytes == 64)
        _mm512_storeu_si512(ptr + idx, __m512i(s.to_register()));
#endif
      else
      {
        // Partial register instead - create a mask which has just enough bits for
        // the partial value to store and defer to the masked store.
        constexpr auto mask = typename _Vec::mask_type(true);
        unchecked_store(s, ptr + idx, _Vec::size, mask, flags);
      }
    };

    chunked_invoke(impl, from);
  }
}

#if defined(__AVX512F__)
/// Load a masked value from memory.
template<vec_type _V, typename _Up, std::size_t _Extent, typename _Flags>
constexpr _V load_masked(x86_avx512_tag, std::span<_Up, _Extent> from, const typename _V::mask_type& mask, _Flags flags)
{
  mandates_for_load(from, _V{}, flags);

  // Note that all of these loads use zero as the default mask value. We can't
  // use the type's actual value because the type of the pointer could be
  // different to the type of the output vec, and therefore be unrepresentable
  // during the load.
  target_overloads call_insn {
    [=](xmm_register<uint8_t> auto, auto m, auto p)  { return _mm_maskz_loadu_epi8(m, p); },
    [=](xmm_register<uint16_t> auto, auto m, auto p) { return _mm_maskz_loadu_epi16(m, p); },
    [=](xmm_register<uint32_t> auto, auto m, auto p) { return _mm_maskz_loadu_epi32(m, p); },
    [=](xmm_register<uint64_t> auto, auto m, auto p) { return _mm_maskz_loadu_epi64(m, p); },

    [=](ymm_register<uint8_t> auto, auto m, auto p)  { return _mm256_maskz_loadu_epi8(m, p); },
    [=](ymm_register<uint16_t> auto, auto m, auto p) { return _mm256_maskz_loadu_epi16(m, p); },
    [=](ymm_register<uint32_t> auto, auto m, auto p) { return _mm256_maskz_loadu_epi32(m, p); },
    [=](ymm_register<uint64_t> auto, auto m, auto p) { return _mm256_maskz_loadu_epi64(m, p); },

    [=](zmm_register<uint8_t>  auto, auto m, auto p) { return _mm512_maskz_loadu_epi8(m, p); },
    [=](zmm_register<uint16_t> auto, auto m, auto p) { return _mm512_maskz_loadu_epi16(m, p); },
    [=](zmm_register<uint32_t> auto, auto m, auto p) { return _mm512_maskz_loadu_epi32(m, p); },
    [=](zmm_register<uint64_t> auto, auto m, auto p) { return _mm512_maskz_loadu_epi64(m, p); },

    //:TODO: DUP all mask bits in a single operation, and only then extract? Cheaper than dupping all bits in small units.
    [=](xmm_register<unsigned __int128> auto, auto m, auto p) { return _mm_maskz_loadu_epi64(dupBits(m), p); },
    [=](ymm_register<unsigned __int128> auto, auto m, auto p) { return _mm256_maskz_loadu_epi64(dupBits(m), p); },
    [=](zmm_register<unsigned __int128> auto, auto m, auto p) { return _mm512_maskz_loadu_epi64(dupBits(m), p); },

    [=](auto unhandled, auto, auto) { static_assert(dependent_false<decltype(unhandled)>, "Unimplemented partial masked load"); return unhandled; }
  };

  using SourceType = rebind_t<_Up, _V>;
  using _C = container_for_type<_Up>;

  auto impl = [=]<typename _CS>(_CS x, auto m, auto idx) {
    // Create a mask which both truncates to the correct number of values and
    // also applies the incoming mask, and then cast to the container type.
    constexpr auto partialMask = typename _CS::mask_type(true);
    const auto tm = m.to_builtin() & partialMask.to_builtin();
    return _CS(call_insn(x, tm, from.data() + idx));
  };

  // Use the mask to limit the bounds of the write if requested by the user.
  auto boundedMask = mask;
  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds<SourceType>("load_masked", from, mask);
  else
    boundedMask &= mask.__mask_from_count(from.size());

  // Load as the container type, and do the conversion as a secondary step.
  // :TODO: Load in the best size to fit a register (i.e., to make for an efficient conversion without spilling into multiple registers).
  auto t = chunked_invoke(impl, rebind_t<_C, _V>(), typename rebind_t<_C, _V>::mask_type(boundedMask));
  return _V(simd_bit_cast<std::remove_cvref_t<_Up>>(t));
}

template<vec_type _V, typename _Up, std::size_t _Extent, typename _Flags>
constexpr auto load(x86_avx512_tag, std::span<_Up, _Extent> from, _Flags flags)
{
  if (contains_flag<flag_unchecked>(flags)) // Unchecked can use the default generic implementation.
    return load<_V>(generic_tag{}, from, flags);
  else                                      // Checked uses a mask to limit the load
    return load_masked<_V>(target, from, typename _V::mask_type(true), flags);
}
#endif

/// @brief Masked gather operation.
/// @tparam _Tp The element type to gather
/// @tparam _Idx the index type
/// @tparam _IdxAbi the ABI of the index type
/// @param data_ptr Pointer to the region of memory from which to gather.
/// @param indexes The indexes to gather from the memory region.
/// @return  A vec of the gathered indexes from the supplied region.
template<typename _Range, std::integral _Idx, typename _IdxAbi, typename... _Flags>
// :TODO: 4/8 byte gathers can be done from either 4/8 byte indexes, so handle
// those scenarios too to avoid having to cast unnecessarily. Also handle
// smaller sizes by reordering the indexes into the gather position, rather than
// converting up and down (which is more expensive).
requires
  std::ranges::contiguous_range<_Range> &&
  std::ranges::sized_range<_Range> &&
  (sizeof(std::ranges::range_value_t<_Range>) <= 8) // Can't handle 128-bit gathers yet.
constexpr auto gather_from(x86_avx2_tag, const _Range& r,
                           const typename basic_vec<_Idx, _IdxAbi>::mask_type& mask,
                           const basic_vec<_Idx, _IdxAbi>& indexes, flags<_Flags...> flags)
{
  using _Tp = std::ranges::range_value_t<_Range>;
  auto data_ptr = std::to_address(r.begin());
  const auto rmax = _Idx(std::min(r.size(), size_t(std::numeric_limits<_Idx>::max())));

  target_overloads impl {

#if defined(__AVX512F__)
    // 32-bit Intel AVX-512
    [=](xmm_register<uint32_t> auto i, auto m) { return _mm_mmask_i32gather_epi32(__m128i(), m.to_register(), i.to_register(), data_ptr, sizeof(_Tp)); },
    [=](ymm_register<uint32_t> auto i, auto m) { return _mm256_mmask_i32gather_epi32(__m256i(), m.to_register(), i.to_register(), data_ptr, sizeof(_Tp)); },
    
    [=](xmm_register<uint64_t> auto i, auto m) { return _mm_mmask_i64gather_epi64(__m128i(), m.to_register(), i.to_register(), data_ptr, sizeof(_Tp)); },
    [=](ymm_register<uint64_t> auto i, auto m) { return _mm256_mmask_i64gather_epi64(__m256i(), m.to_register(), i.to_register(), data_ptr, sizeof(_Tp)); },

    [=](zmm_register<uint32_t> auto i, auto m) { return _mm512_mask_i32gather_epi32(__m512i(), m.to_register(), i.to_register(), data_ptr, sizeof(_Tp)); },
    [=](zmm_register<uint64_t> auto i, auto m) { return _mm512_mask_i64gather_epi64(__m512(), m.to_register(), i.to_register(), data_ptr, sizeof(_Tp)); },

#elif defined(__AVX2__)
    // Note the explicit growth of the mask to the correct register size to
    // avoid inadvertently leaving uninitialised elements active.

    // 32-bit Intel AVX2
    [=](xmm_register<uint32_t> auto i, auto m)
      { return _mm_mask_i32gather_epi32(__m128i(), data_ptr, i.to_register(), grow<4>(m).to_register(), sizeof(_Tp)); },
    [=](ymm_register<uint32_t> auto i, auto m)
      { return _mm256_mask_i32gather_epi32(__m256i(), data_ptr, i.to_register(), grow<8>(m).to_register(), sizeof(_Tp)); },

    // 64-bit Intel AVX2
    [=](xmm_register<uint64_t> auto i, auto m)
      { return _mm_mask_i64gather_epi64(__m128i(), data_ptr, i.to_register(), grow<2>(m).to_register(), sizeof(_Tp)); },
    [=](ymm_register<uint64_t> auto i, auto m)
      { return _mm256_mask_i64gather_epi64(__m256i(), data_ptr, i.to_register(), grow<4>(m).to_register(), sizeof(_Tp)); },

#endif

    [=](auto unhandled, auto) { static_assert(dependent_false<decltype(unhandled)>); }
  };

  // Convert the indexes to at least 32-bit, since no instruction exists
  // operating at a smaller granularity.
  constexpr auto idxNumBytes = std::max({sizeof(uint32_t), sizeof(_Tp), sizeof(_Idx)});
  using _IW = container_for_num_bytes<idxNumBytes>;

  auto wrapper = [=]<typename _VecIdx>(_VecIdx idx, auto m) {
    // Bump the indexes and mask up to the preferred size if necessary.
    using RetypedIndex = vec<_IW, _VecIdx::size>;
    auto ri = RetypedIndex(idx);

    // Note that the mask is actively resized to zero out unwanted positions.
    // Converting to the register in the builtin will merely leave unused
    // positions uninitialized, which might make them active with an invalid
    // index.
    auto rm = typename RetypedIndex::mask_type(m);

    // Add a bounds check if necessary.
    if constexpr (!contains_flag<flag_unchecked>(flags))
      rm = rm && (ri < rmax);

    // Do the gather, reading back elements which are at least as big as the required type, and then downcasting to the types size if necessary.
    auto t = RetypedIndex(impl(ri, rm));
    auto r = vec<container_for_type<_Tp>, _VecIdx::size>(t);

    // Return the values in their proper type form.
    return simd_bit_cast<_Tp>(r);
  };

  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds("gather", indexes, rmax, mask);

  return chunked_invoke<int(vec<_IW>::size)>(wrapper, indexes, mask);
}

template<detail::contiguous_sized_range _Range, vec_integral _Idx, typename... _Flags>
constexpr auto gather_from(x86_avx2_tag, const _Range& r,
                           const _Idx& indexes, flags<_Flags...> flags)
{
  // Forward to the normal gather. All gathers are masked anyway (i.e., the
  // compiler will create a mask anyway), so this doesn't make much difference.
  constexpr auto m = typename _Idx::mask_type(true);
  return partial_gather_from(r, m, indexes, flags);
}

#if defined(__AVX512F__)

inline auto scatterImpl =
  []<typename _Tp, typename _VT, typename _IT>(basic_vec<_Tp, _VT> v,
                                               basic_vec<uint32_t, _IT> idx, auto ptr, uint64_t m)
{
  constexpr uint64_t justMaskBits = (1ULL << basic_vec<_Tp, _VT>::size()) - 1;
  const uint64_t nm = justMaskBits & m;

  constexpr bool isTp32 = (sizeof(_Tp) == 4);

  if constexpr (xmm_register<basic_vec<_Tp, _VT>>)
  {
    if constexpr (isTp32) _mm_mask_i32scatter_epi32(ptr, nm, idx.to_register(), v.to_register(), 4);
    else                  _mm_mask_i32scatter_epi64(ptr, nm, idx.to_register(), v.to_register(), 8);
  }
  else if constexpr (ymm_register<basic_vec<_Tp, _VT>>)
  {
    if constexpr (isTp32) _mm256_mask_i32scatter_epi32(ptr, nm, idx.to_register(), v.to_register(), 4);
    else                  _mm256_mask_i32scatter_epi64(ptr, nm, idx.to_register(), v.to_register(), 8);
  }
  else
  {
    if constexpr (isTp32) _mm512_mask_i32scatter_epi32(ptr, nm, idx.to_register(), v.to_register(), 4);
    else                  _mm512_mask_i32scatter_epi64(ptr, nm, idx.to_register(), v.to_register(), 8);
  }
};

/// Scatter the contents of a vec value into the supplied memory at the given index positions.
template<typename _Range, typename _Tp, typename _TpAbi, typename _Idx, typename _IdxAbi, typename... _Flags>
requires (sizeof(_Tp) == 4 || sizeof(_Tp) == 8) // Only 32/64-bit writes are natively supported in x86. For smaller or larger elements the
                                                // generic loop implementation is faster.
constexpr void
scatter_to(x86_avx512_tag, const basic_vec<_Tp, _TpAbi>& values, _Range&& r,
           const basic_vec<_Idx, _IdxAbi>& indexes, flags<_Flags...> flags)
{
  auto ptr = std::to_address(r.begin());
  const auto rmax = _Idx(std::min(r.size(), size_t(std::numeric_limits<_Idx>::max())));

  // Bind extra pointer and default mask parameters to the scatter lambda. Note
  // that a range check is added if necessary too.
  auto impl = [=](auto v, auto i) {
    if constexpr (contains_flag<detail::flag_unchecked>(flags))
      scatterImpl(v, i, ptr, ~0ULL);
    else
      scatterImpl(v, i, ptr, (i < rmax).to_register());
  };

  if constexpr (contains_flag<flag_unchecked>(flags))
   checkStaticMemoryBounds("scatter", indexes, rmax);

  // All scatter instructions can be made to use 32-bit indexes.
  // :TODO: A special case might be 64-bit values using 64-bit indexes
  auto indexAs32 = vec<uint32_t, basic_vec<_Idx, _IdxAbi>::size>(indexes);
  chunked_invoke<(int)basic_vec<_Tp>::size>(impl, vec_as_container(values), indexAs32);
}

template<typename _Range, typename _Tp, typename _TpAbi, typename _Idx, typename _IdxAbi, typename... _Flags>
requires (sizeof(_Tp) == 4 || sizeof(_Tp) == 8) // Only 32/64-bit writes are natively supported in x86. For smaller or larger elements the
                                                 // generic loop implementation is faster.
constexpr void
scatter_to(x86_avx512_tag, const basic_vec<_Tp, _TpAbi>& values, _Range&& r,
           const typename basic_vec<_Idx, _IdxAbi>::mask_type& mask,
           const basic_vec<_Idx, _IdxAbi>& indexes, flags<_Flags...> flags)
{
  auto ptr = std::to_address(r.begin());

  // Range check if required.
  const auto rmax = _Idx(std::min(r.size(), size_t(std::numeric_limits<_Idx>::max())));
  auto rangeCheckedMask =
    contains_flag<detail::flag_unchecked>(flags) ? mask : (indexes < rmax) && mask;

  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds("scatter", indexes, rmax, mask);

  // Bind an extra pointer parameter to the scatter lambda
  auto impl = [=](auto v, auto i, auto m) { scatterImpl(v, i, ptr, m.to_ullong()); };

  // All scatter instructions can be made to use 32-bit indexes.
  // :TODO: A special case might be 64-bit values using 64-bit indexes
  auto indexAs32 = vec<uint32_t, basic_vec<_Idx, _IdxAbi>::size>(indexes);
  chunked_invoke<(int)basic_vec<_Tp>::size()>(impl, vec_as_container(values), indexAs32, rangeCheckedMask);
}
#endif

// x86 binary operators to generate a mask using the underlying compiler builtin type.
template<typename _Tp, typename _Abi, mask_operator _Op> // Concept - binary operator of some type. :TODO: differentiate between binary and unary?
constexpr auto builtin_operator(x86_avx512_tag, basic_vec<_Tp, _Abi> x, basic_vec<_Tp, _Abi> y, _Op)
{
  using binop_type = std::remove_cvref_t<_Op>;
  using value_type = std::remove_cvref_t<_Tp>;

  // Dispatch to the correct operation.
  constexpr bool is_int = std::is_integral_v<value_type> || std::is_enum_v<value_type>;
  constexpr auto EQ_OP = is_int ? _MM_CMPINT_EQ : _CMP_EQ_OQ;
  constexpr auto NEQ_OP = is_int ? _MM_CMPINT_NE : _CMP_NEQ_OQ;
  constexpr auto LT_OP = is_int ? _MM_CMPINT_LT : _CMP_LT_OQ;
  constexpr auto LE_OP = is_int ? _MM_CMPINT_LE : _CMP_LE_OQ;
  constexpr auto GT_OP = is_int ? _MM_CMPINT_NLE : _CMP_GT_OQ;
  constexpr auto GE_OP = is_int ? _MM_CMPINT_NLT : _CMP_GE_OQ;

  constexpr auto op = (std::is_same_v<binop_type, std::equal_to<>>) ? EQ_OP :
                      (std::is_same_v<binop_type, std::not_equal_to<>>) ? NEQ_OP :
                      (std::is_same_v<binop_type, std::less<>>) ? LT_OP :
                      (std::is_same_v<binop_type, std::less_equal<>>) ? LE_OP :
                      (std::is_same_v<binop_type, std::greater<>>) ? GT_OP :
                      GE_OP;

  static_assert(op != GE_OP || std::is_same_v<binop_type, std::greater_equal<>>, "Unknown binary operation");

  auto impl = [=](auto lhs, auto rhs) {

    constexpr bool isSigned = std::is_signed_v<value_type>;

    auto lhsr = lhs.to_register();
    auto rhsr = rhs.to_register();

    if constexpr (is_int)
    {
      if constexpr (sizeof(value_type) == 1) {
        if constexpr      (sizeof(lhs) <= 16) return isSigned ? _mm_cmp_epi8_mask(lhsr, rhsr, op)    : _mm_cmp_epu8_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 32) return isSigned ? _mm256_cmp_epi8_mask(lhsr, rhsr, op) : _mm256_cmp_epu8_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 64) return isSigned ? _mm512_cmp_epi8_mask(lhsr, rhsr, op) : _mm512_cmp_epu8_mask(lhsr, rhsr, op);
      }
      else if constexpr (sizeof(value_type) == 2) {
        if constexpr      (sizeof(lhs) <= 16) return isSigned ? _mm_cmp_epi16_mask(lhsr, rhsr, op)    : _mm_cmp_epu16_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 32) return isSigned ? _mm256_cmp_epi16_mask(lhsr, rhsr, op) : _mm256_cmp_epu16_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 64) return isSigned ? _mm512_cmp_epi16_mask(lhsr, rhsr, op) : _mm512_cmp_epu16_mask(lhsr, rhsr, op);
      }
      else if constexpr (sizeof(value_type) == 4) {
        if constexpr      (sizeof(lhs) <= 16) return isSigned ? _mm_cmp_epi32_mask(lhsr, rhsr, op)    : _mm_cmp_epu32_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 32) return isSigned ? _mm256_cmp_epi32_mask(lhsr, rhsr, op) : _mm256_cmp_epu32_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 64) return isSigned ? _mm512_cmp_epi32_mask(lhsr, rhsr, op) : _mm512_cmp_epu32_mask(lhsr, rhsr, op);
      }
      if constexpr (sizeof(value_type) == 8) {
        if constexpr      (sizeof(lhs) <= 16) return isSigned ? _mm_cmp_epi64_mask(lhsr, rhsr, op)    : _mm_cmp_epu64_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 32) return isSigned ? _mm256_cmp_epi64_mask(lhsr, rhsr, op) : _mm256_cmp_epu64_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 64) return isSigned ? _mm512_cmp_epi64_mask(lhsr, rhsr, op) : _mm512_cmp_epu64_mask(lhsr, rhsr, op);
      }
    }
    else if constexpr (is_fp16_v<value_type>)
    {
      #if defined (__AVX512FP16__)
        if constexpr      (sizeof(lhs) <= 16) return _mm_cmp_ph_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 32) return _mm256_cmp_ph_mask(lhsr, rhsr, op);
        else if constexpr (sizeof(lhs) <= 64) return _mm512_cmp_ph_mask(lhsr, rhsr, op);
      #else
        static_assert(dependent_false<decltype(lhs)>, "Emulated support for _Float16 handled below");
      #endif
    }
    else if constexpr (std::is_same_v<value_type, float>)
    {
      if constexpr      (sizeof(lhs) <= 16) return _mm_cmp_ps_mask(lhsr, rhsr, op);
      else if constexpr (sizeof(lhs) <= 32) return _mm256_cmp_ps_mask(lhsr, rhsr, op);
      else if constexpr (sizeof(lhs) <= 64) return _mm512_cmp_ps_mask(lhsr, rhsr, op);
    }
    else if constexpr (std::is_same_v<value_type, double>)
    {

      if constexpr      (sizeof(lhs) <= 16) return _mm_cmp_pd_mask(lhsr, rhsr, op);
      else if constexpr (sizeof(lhs) <= 32) return _mm256_cmp_pd_mask(lhsr, rhsr, op);
      else if constexpr (sizeof(lhs) <= 64) return _mm512_cmp_pd_mask(lhsr, rhsr, op);
    }
    else
      static_assert(dependent_false<decltype(lhs)>);
  };

  // Provide a wrapper to return the correct type of mask from whatever the
  // intrinsic returns above. It is easiest to do that once here, than to have
  // to make every intrinsic above do it. Same with the data member - unwrap
  // once here.
  if constexpr (is_fp16_v<_Tp> && !hasAvxFp16)
  {
    // When FP16 comparison is needed in emulation, do the comparison using their
    // FP32 equivalents. All FP16 values can be converted to FP32 while
    // retaining their relationships with each other.
    auto m32 = builtin_operator(x86_avx512_tag{}, rebind_cast<float>(x), rebind_cast<float>(y), _Op{});
    return basic_mask<sizeof(_Tp), _Abi>(m32);
  }
  else
  {
    // Wrapper to put the correct return type on a plain integer from an intrinsic.
    auto wrapper = [=](auto l, auto r) { return mask<_Tp, decltype(l.size)::value>(impl(l, r)); };
    return chunked_invoke(wrapper, x, y);
  }
}

/// @brief Implement equality/inequality for complex values. In all cases the
/// underlying elements are compared (e.g., float for std::complex<float>) using
/// an appropriate operator, and then adjacent bits in the comparison are
/// combined together using AND/OR operation. For example, in equality, the two
/// individual elements in each complex values are first compared for equality,
/// and then the overall bit for the single complex element is created by ANDing
/// the two bits together (i.e., a complex number is only equal if both of its
/// individual elements are equal).
/// @tparam _Tp The type of the elements to compare.
/// @tparam _Op The operation to perform. Must be equal_to or not_equal_to.
/// @tparam _Np The number of complex elements in the vec value.
/// @param lhs The left-hand operand to compare
/// @param rhs The right-hand operand to compare
/// @return A compact bit mask with one element for each complex element.
/// \internal
template<typename _Tp, typename _Abi, mask_operator _Op> // :TODO: Or specifically equal-not-equal?
requires (detail::is_floating_point_v<_Tp>)
constexpr auto builtin_operator(x86_avx512_tag, basic_vec<std::complex<_Tp>, _Abi> lhs, basic_vec<std::complex<_Tp>, _Abi> rhs, _Op cmp_op)
{
  using binop_type = std::remove_cvref_t<_Op>;
  constexpr bool isEqualOp = std::is_same_v<binop_type, std::equal_to<>>;
  constexpr bool isNotEqualOp = std::is_same_v<binop_type, std::not_equal_to<>>;
  static_assert(isEqualOp || isNotEqualOp, "Complex values only allow equality comparisons");

  /// Given a mask of up to 32 bits, reduce it down to 16 bits by either AND'ing or OR'ing adjacent bits.
  const auto reduceBitPair = [](auto bits) {
    if constexpr (isEqualOp) // AND bits
      return _mm256_cmp_epi16_mask(_mm256_movm_epi8(bits), _mm256_set1_epi16(-1), _MM_CMPINT_EQ);
    else // OR bits
      return _mm256_cmp_epi16_mask(_mm256_movm_epi8(bits), __m256i(), _MM_CMPINT_NE);
  };

  auto impl = [=]<typename _Vec>(_Vec x, _Vec y)
  {
    const auto bits = builtin_operator(x86_avx512_tag{}, simd_bit_cast<_Tp>(x), simd_bit_cast<_Tp>(y), cmp_op);
    auto rb = reduceBitPair(bits.to_builtin());
    return typename _Vec::mask_type(rb);
  };

  return chunked_invoke(impl, lhs, rhs);
}

/// Constants which specify to the compiler intrinsic which FP classification operation to perform
// :COMPILER: Perhaps the constants should be defined in immintrin?
enum X86_FP_CLASSIFY
{
  QUIET_NAN = 0x01,
  POS_ZERO = 0x02,
  NEG_ZERO = 0x04,
  POS_INF = 0x08,
  NEG_INF = 0x10,
  DENORM = 0x20,
  NEG = 0x40,
  SIGNAL_NAN = 0x80
};

/// @brief Classify the floating-point vec values using the fpclassify intrinsic.
/// @tparam _Tp The type of floating-point value to classify
/// @tparam _Mp The type of mask element to generate for the given floating-point value
/// @tparam _Rp The result mask to generate
/// @tparam _Op The set of immediate values for fp_classify to query.
/// @tparam _Np The number of vec elements.
/// @param value A set of floating-point vec values
/// @return A compact mask which has a bit set if the corresponding element in
/// the input meets the classification requirements, and false otherwise.
template<int _Op, typename _Tp, typename _Abi>
inline auto x86_fp_classify_mask(const basic_vec<_Tp, _Abi>& value) {

  target_overloads impl {
#if defined (__AVX512FP16__)
    // GCC has a bug in its implementation of the fpclass intrinsic so use the
    // underlying builtin directly. Clang doesn't care which implementation is
    // used so there is no need to have gcc-specific conditional code.
    [=]<xmm_register<_Float16> _Vec>(_Vec v) -> __mmask8 {
      return __builtin_ia32_fpclassph128_mask(v.to_register(), _Op, (__mmask64(1) << _Vec::size) - 1);
    },
    [=]<ymm_register<_Float16> _Vec>(_Vec v) -> __mmask16 {
      return __builtin_ia32_fpclassph256_mask(v.to_register(), _Op, (__mmask64(1) << _Vec::size) - 1);
    },
    [=]<zmm_register<_Float16> _Vec>(_Vec v) -> __mmask32 {
      return __builtin_ia32_fpclassph512_mask(v.to_register(), _Op, (__mmask64(1) << _Vec::size) - 1);
    },
#endif
    [=](xmm_register<float> auto v) { return _mm_fpclass_ps_mask(v.to_register(), _Op); },
    [=](ymm_register<float> auto v) { return _mm256_fpclass_ps_mask(v.to_register(), _Op); },
  
    [=](xmm_register<double> auto v) { return _mm_fpclass_pd_mask(v.to_register(), _Op); },
    [=](ymm_register<double> auto v) { return _mm256_fpclass_pd_mask(v.to_register(), _Op); },

    [=](zmm_register<float> auto v) { return _mm512_fpclass_ps_mask(v.to_register(), _Op); },
    [=](zmm_register<double> auto v) { return _mm512_fpclass_pd_mask(v.to_register(), _Op); },

    [=](auto unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return 0; }
  };

  // Put a suitable mask around the bare integer from impl and then invoke.
  auto wrapper = [=](auto _vec) {
    auto dwt = impl(_vec);
    return mask<_Tp, decltype(_vec.size)::value>(dwt);
  };
  return chunked_invoke(wrapper, value);
}

/// @brief Detect whether native classification support is available for the
/// given type. Emulated FP16 has no such support, but native FP16 does.
/// @tparam _Tp The type to query for classification support.
template<typename _Tp> constexpr bool has_native_support_for_classify =
  detail::is_floating_point_v<_Tp> && (hasAvxFp16 || !is_fp16_v<_Tp>);

/// x86 floating-point classification function.
template<typename _Tp, typename _Abi>
requires (has_native_support_for_classify<_Tp>)
constexpr typename basic_vec<_Tp, _Abi>::mask_type
isinf(x86_avx512_tag, const basic_vec<_Tp, _Abi>& v)
{
  return x86_fp_classify_mask<X86_FP_CLASSIFY::POS_INF | X86_FP_CLASSIFY::NEG_INF>(v);
}

template<typename _Tp, typename _Abi>
requires (has_native_support_for_classify<_Tp>)
constexpr typename basic_vec<_Tp, _Abi>::mask_type
isnan(x86_avx512_tag, const basic_vec<_Tp, _Abi>& v)
{
  return x86_fp_classify_mask<X86_FP_CLASSIFY::QUIET_NAN | X86_FP_CLASSIFY::SIGNAL_NAN>(v);
}

template<typename _Tp, typename _Abi>
requires (has_native_support_for_classify<_Tp>)
constexpr typename basic_vec<_Tp, _Abi>::mask_type
isfinite(x86_avx512_tag, const basic_vec<_Tp, _Abi>& v)
{
  constexpr int invalidClasses = X86_FP_CLASSIFY::QUIET_NAN | X86_FP_CLASSIFY::SIGNAL_NAN | X86_FP_CLASSIFY::POS_INF | X86_FP_CLASSIFY::NEG_INF;
  return !x86_fp_classify_mask<invalidClasses>(v);
}

template<typename _Tp, typename _Abi>
requires (has_native_support_for_classify<_Tp>)
constexpr typename basic_vec<_Tp, _Abi>::mask_type
isnormal(x86_avx512_tag, const basic_vec<_Tp, _Abi>& v)
{
  constexpr int invalidClasses =
    X86_FP_CLASSIFY::QUIET_NAN | X86_FP_CLASSIFY::SIGNAL_NAN | X86_FP_CLASSIFY::POS_INF | X86_FP_CLASSIFY::NEG_INF |
    X86_FP_CLASSIFY::POS_ZERO | X86_FP_CLASSIFY::NEG_ZERO | X86_FP_CLASSIFY::DENORM; 
  return !x86_fp_classify_mask<invalidClasses>(v);
}

#if defined(__AVX512F__)
template<std::integral _Up, std::integral _Tp, typename _Abi>
requires (std::signed_integral<_Up> == std::signed_integral<_Tp> &&    // Only same-sign conversions have ISA support
          sizeof(_Up) < sizeof(_Tp))                                   // Type must be getting smaller. Same size or bigger can be handled by generic cast
constexpr auto
saturating_cast(x86_avx512_tag, const basic_vec<_Tp, _Abi>& value) noexcept
{
  target_overloads impl {
    // S16 -> S8
    [=](xmm_register<std::int16_t> auto in, int8_t) { return _mm_cvtsepi16_epi8(in.to_register()); },
    [=](ymm_register<std::int16_t> auto in, int8_t) { return _mm256_cvtsepi16_epi8(in.to_register()); },
    [=](zmm_register<std::int16_t> auto in, int8_t) { return _mm512_cvtsepi16_epi8(in.to_register()); },

    // U16 -> U8
    [=](xmm_register<std::uint16_t> auto in, uint8_t) { return _mm_cvtusepi16_epi8(in.to_register()); },
    [=](ymm_register<std::uint16_t> auto in, uint8_t) { return _mm256_cvtusepi16_epi8(in.to_register()); },
    [=](zmm_register<std::uint16_t> auto in, uint8_t) { return _mm512_cvtusepi16_epi8(in.to_register()); },

    // S32 -> S8
    [=](xmm_register<std::int32_t> auto in, int8_t) { return _mm_cvtsepi32_epi8(in.to_register()); },
    [=](ymm_register<std::int32_t> auto in, int8_t) { return _mm256_cvtsepi32_epi8(in.to_register()); },
    [=](zmm_register<std::int32_t> auto in, int8_t) { return _mm512_cvtsepi32_epi8(in.to_register()); },

    // U32 -> U8
    [=](xmm_register<std::uint32_t> auto in, uint8_t) { return _mm_cvtusepi32_epi8(in.to_register()); },
    [=](ymm_register<std::uint32_t> auto in, uint8_t) { return _mm256_cvtusepi32_epi8(in.to_register()); },
    [=](zmm_register<std::uint32_t> auto in, uint8_t) { return _mm512_cvtusepi32_epi8(in.to_register()); },

    // S32 -> S16
    [=](xmm_register<std::int32_t> auto in, int16_t) { return _mm_cvtsepi32_epi16(in.to_register()); },
    [=](ymm_register<std::int32_t> auto in, int16_t) { return _mm256_cvtsepi32_epi16(in.to_register()); },
    [=](zmm_register<std::int32_t> auto in, int16_t) { return _mm512_cvtsepi32_epi16(in.to_register()); },

    // U32 -> U16
    [=](xmm_register<std::uint32_t> auto in, uint16_t) { return _mm_cvtusepi32_epi16(in.to_register()); },
    [=](ymm_register<std::uint32_t> auto in, uint16_t) { return _mm256_cvtusepi32_epi16(in.to_register()); },
    [=](zmm_register<std::uint32_t> auto in, uint16_t) { return _mm512_cvtusepi32_epi16(in.to_register()); },

    // S64 -> S8
    [=](xmm_register<std::int64_t> auto in, int8_t) { return _mm_cvtsepi64_epi8(in.to_register()); },
    [=](ymm_register<std::int64_t> auto in, int8_t) { return _mm256_cvtsepi64_epi8(in.to_register()); },
    [=](zmm_register<std::int64_t> auto in, int8_t) { return _mm512_cvtsepi64_epi8(in.to_register()); },

    // U64 -> U8
    [=](xmm_register<std::uint64_t> auto in, uint8_t) { return _mm_cvtusepi64_epi8(in.to_register()); },
    [=](ymm_register<std::uint64_t> auto in, uint8_t) { return _mm256_cvtusepi64_epi8(in.to_register()); },
    [=](zmm_register<std::uint64_t> auto in, uint8_t) { return _mm512_cvtusepi64_epi8(in.to_register()); },

    // S64 -> S16
    [=](xmm_register<std::int64_t> auto in, int16_t) { return _mm_cvtsepi64_epi16(in.to_register()); },
    [=](ymm_register<std::int64_t> auto in, int16_t) { return _mm256_cvtsepi64_epi16(in.to_register()); },
    [=](zmm_register<std::int64_t> auto in, int16_t) { return _mm512_cvtsepi64_epi16(in.to_register()); },

    // U64 -> U16
    [=](xmm_register<std::uint64_t> auto in, uint16_t) { return _mm_cvtusepi64_epi16(in.to_register()); },
    [=](ymm_register<std::uint64_t> auto in, uint16_t) { return _mm256_cvtusepi64_epi16(in.to_register()); },
    [=](zmm_register<std::uint64_t> auto in, uint16_t) { return _mm512_cvtusepi64_epi16(in.to_register()); },

    // S64 -> S32
    [=](xmm_register<std::int64_t> auto in, int32_t) { return _mm_cvtsepi64_epi32(in.to_register()); },
    [=](ymm_register<std::int64_t> auto in, int32_t) { return _mm256_cvtsepi64_epi32(in.to_register()); },
    [=](zmm_register<std::int64_t> auto in, int32_t) { return _mm512_cvtsepi64_epi32(in.to_register()); },

    // U64 -> U32
    [=](xmm_register<std::uint64_t> auto in, uint32_t) { return _mm_cvtusepi64_epi32(in.to_register()); },
    [=](ymm_register<std::uint64_t> auto in, uint32_t) { return _mm256_cvtusepi64_epi32(in.to_register()); },
    [=](zmm_register<std::uint64_t> auto in, uint32_t) { return _mm512_cvtusepi64_epi32(in.to_register()); },

    [=](auto unhandled, auto out)
      { static_assert(dependent_false<std::pair<decltype(unhandled), decltype(out)>>, "No target overload"); return out;}
  };

  auto wrapper = [=]<typename _VT, typename _VAbi>(basic_vec<_VT, _VAbi> v) {
    auto r = impl(v, _Up()); // Pass in dummy parameter to be able to use overloading.
    return vec<_Up, basic_vec<_VT, _VAbi>::size>(r);
  };
  return chunked_invoke(wrapper, value);
}
#endif

/// In-lane (128-bit) byte shuffles
///@{
inline __m128i shuffle_epi8(__m128i values, __m128i indexes) { return _mm_shuffle_epi8(values, indexes); }

inline __m256i shuffle_epi8(__m256i values, __m256i indexes) {
#if defined (__AVX2__)
  return _mm256_shuffle_epi8(values, indexes);
#else
  // Intel AVX doesn't have a 256-bit shuffle so handle it using a pair of
  // 128-bit shuffles instead. Note that users of this function might consider
  // rewriting the call to improve efficiency instead.
  const auto lowerShuf = _mm_shuffle_epi8(_mm256_castsi256_si128(values), _mm256_castsi256_si128(indexes));
  const auto upperShuf = _mm_shuffle_epi8(_mm256_extractf128_si256(values, 1), _mm256_extractf128_si256(indexes, 1));
  return _mm256_insertf128_si256(_mm256_castsi128_si256(lowerShuf), upperShuf, 1);
#endif
}

#if defined(__AVX512F__)
inline __m512i shuffle_epi8(__m512i values, __m512i indexes) { return _mm512_shuffle_epi8(values, indexes); }
#endif

/// Perform a shuffle_epi8 from a table of bytes created from the given
/// generator. This is a basic building block for many bit query operations
/// (e.g., popcount, bit reverse). It can be implemented efficiently by using
/// the lane-wise shuffle_epi8 instruction, where the LUT is duplicated into
/// each lane. Ths indexing behaves exactly like a shuffle_epi8 (i.e., use the
/// bottom 4 bits as a lane-wise index, zeroing if the index MSB is set). The
/// name is chosen to reflect this specialist use (i.e., this is not a general
/// 4-bit lut).
template<typename _Gen, typename _Abi>
constexpr basic_vec<std::uint8_t, _Abi>
shuffle_epi8_lut(_Gen, basic_vec<std::uint8_t, _Abi> indexes)
{
  auto rawIndexes = indexes.to_register();

  // Build a LUT which contains as many lanes as the indexes.
  constexpr auto lut = vec<uint8_t, sizeof(rawIndexes)>([](auto idx) { return _Gen{}(idx & 0xF); });

  // Use the fast in-lane byte shuffle.
  return basic_vec<std::uint8_t, _Abi>(shuffle_epi8(lut.to_register(), rawIndexes));
}

///}@

/// @brief Compute the number of 1 bits in each vec element
/// @param x The value to query
/// @return A vec containing the number of 1 bits found in each respective element.
/// @internal
template<typename _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> popcount(x86_sse_tag, const basic_vec<_Tp, _Abi>& x) noexcept {

  const auto bitsInByte = []<typename _Vec>(_Vec piece) -> _Vec {
    const auto bytes = simd_bit_cast<std::uint8_t>(piece);

    // Use a lookup table to compute the population count for 4-bits at a time
    // for lower and upper halves of a byte, then add together.
    constexpr auto gen = [](auto idx) -> std::uint8_t { return std::popcount(std::uint8_t(idx & 0xF)); };

    const auto lowPopCount = shuffle_epi8_lut(gen, bytes & cw<0xF>);
    const auto highPopCount = shuffle_epi8_lut(gen, bytes >> 4);

    return simd_bit_cast<_Tp>(lowPopCount + highPopCount);
  };

  // Compute the population counts for the individual bytes.
  auto pc = chunked_invoke(bitsInByte, x);

  if constexpr (sizeof(_Tp) == 1) return pc;

  // Sum adjacent bytes for larger values.
  if constexpr (sizeof(_Tp) == 8) pc += (pc >> 32);
  if constexpr (sizeof(_Tp) >= 4) pc += (pc >> 16);
  if constexpr (sizeof(_Tp) >= 2) pc += (pc >> 8);

  // Only the lowest byte of each element contains the correct count. The other bytes contain left-over debris.
  return pc & cw<0xFF>;
}

#if defined(__AVX512VPOPCNTDQ__)
// Sunny Cove has native population count instructions.
template<typename _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> popcount(x86_avxsnc_tag, const basic_vec<_Tp, _Abi>& x) noexcept {
  target_overloads impl {
    [=](xmm_register<std::uint8_t>  auto v) {  return _mm_popcnt_epi8(v.to_register()); },
    [=](xmm_register<std::uint16_t> auto v) {  return _mm_popcnt_epi16(v.to_register()); },
    [=](xmm_register<std::uint32_t> auto v) {  return _mm_popcnt_epi32(v.to_register()); },
    [=](xmm_register<std::uint64_t> auto v) {  return _mm_popcnt_epi64(v.to_register()); },

    [=](ymm_register<std::uint8_t>  auto v) {  return _mm256_popcnt_epi8(v.to_register()); },
    [=](ymm_register<std::uint16_t> auto v) {  return _mm256_popcnt_epi16(v.to_register()); },
    [=](ymm_register<std::uint32_t> auto v) {  return _mm256_popcnt_epi32(v.to_register()); },
    [=](ymm_register<std::uint64_t> auto v) {  return _mm256_popcnt_epi64(v.to_register()); },

    [=](zmm_register<std::uint8_t>  auto v) {  return _mm512_popcnt_epi8(v.to_register()); },
    [=](zmm_register<std::uint16_t> auto v) {  return _mm512_popcnt_epi16(v.to_register()); },
    [=](zmm_register<std::uint32_t> auto v) {  return _mm512_popcnt_epi32(v.to_register()); },
    [=](zmm_register<std::uint64_t> auto v) {  return _mm512_popcnt_epi64(v.to_register()); },

    [=](auto unhandled) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  // Call inside wrapper which converts intrinsic into vec value.
  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, x);
}
#endif // __AVX512VPOPCNTDQ__

/// @brief Query whether each element has exactly one bit set.
/// @param x The value to query
/// @return A simd::mask which has an element set to true if the element has exactly one set bit.
/// @internal
template<typename _Tp, typename _Abi>
constexpr auto has_single_bit(x86_avxsnc_tag, const basic_vec<_Tp, _Abi>& x) noexcept {
  return popcount(x) == cw<1>;
}

/// @brief Count the number of leading zero bits in each element
/// @param x The input value
/// @return A vec value in which each element contains the number of 1 bits in each respective input element.
/// @internal
template<std::unsigned_integral _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> clz(x86_sse_tag, const basic_vec<_Tp, _Abi>& x) noexcept {

  // Utility to compute clz for individual bytes using table lookup.
  const auto clzByte = []<typename _Vec>(_Vec piece) -> _Vec
  {
    const auto bytes = simd_bit_cast<std::uint8_t>(piece);

    // Compute the 4-bit low and high clzs for each 8-bit value. Note that
    // shuffle uses a dirty index where the high bit means zero the output, so
    // the top bits must normally be masked away. There is no need to handle
    // that special case though because if the MSB is set then the select below
    // will remove it anyway. Note also that constant 4 is added to the low clz
    // to account for it being the bottom 4 bits to start with. The 4 could have
    // been added to the result of the clz but by putting it on the values to
    // lookup the compiler can hoist it.
    const auto lowClz =
      shuffle_epi8_lut([](auto idx) -> std::uint8_t { return std::countl_zero(std::uint8_t(idx & 0xF)); }, bytes & uint8_t(0xF));
    const auto highClz =
      shuffle_epi8_lut([](auto idx) -> std::uint8_t { return std::countl_zero(std::uint8_t(idx & 0xF)) - 4; }, bytes >> uint8_t(4));

    // Select which clz actually applies here.
    return simd_bit_cast<_Tp>(select(bytes > uint8_t(15), highClz, lowClz));
  };

  constexpr auto numElementBytes = sizeof(_Tp);

  if constexpr (numElementBytes == 1)
    return chunked_invoke(clzByte, x);
  else if constexpr (numElementBytes == 2)
  {
    // Use the 8-bit variant to build the 16-bit variant.
    const auto clz8 = chunked_invoke(clzByte, x);
    // If there are any bits in the upper byte use the top byte, otherwise use
    // the bottom byte with an extra 8 added to account for the empty top bits.
    return select(x > cw<0xFF>, clz8 >> 8, (clz8 & cw<0xFF>) + cw<8>);
  }
  else if constexpr (numElementBytes == 4)
  {
    // Use the FP conversion trick described in Hackers Delight, page 82.
    // Floating-point format essentially stores the index of the first
    // leading bit in the mantissa so this can be extracted after converting
    // the number in question to FP. Care must be taken with rounding and
    // the special case of 0. Note also that a conversion from unsigned to
    // float is unsupported in Intel AVX and the compiler's synthesis is quite
    // expensive. It is cheaper to pretend it is signed int and test for the
    // special case of the MSB being set and blend in the correct answer
    // than do the conversion.
    const auto keepTopBit = rebind_cast<std::int32_t>(x & ~(x >> 1));
    const auto fp = rebind_cast<float>(keepTopBit) + 0.5f;

    const auto t = (cw<158> - (simd_bit_cast<std::uint32_t>(fp) >> 23));
    const auto mt = min(t, basic_vec<_Tp, _Abi>(cw<32>));

    return select((x & 0x80000000) == cw<0>, mt, basic_vec<_Tp, _Abi>());
  }
  else if constexpr (numElementBytes == 8)
  {
    // Use the 32-bit variant to build the 64-bit variant.
    const auto clz32 = countl_zero(simd_bit_cast<std::uint32_t>(x));

    // Treat the individual clz values as parts of a 64-bit value.
    const auto tas64 = simd_bit_cast<std::uint64_t>(clz32);

    // If there are any bits in the upper word use that count. If the top word
    // is empty use whatever the bottom word's count is but compensate for the
    // empty top word by adding another 32.
    return select(x > 0xFFFFFFFF, tas64 >> 32, (tas64 & 0xFFFFFFFF) + cw<32>);

    // This is an alternative for machines which have fast std::uint64_t -> double conversion (which Intel AVX or Intel SSE don't have).
    // const auto keepTopBit = v & ~(v >> 1);
    // const auto fp = rebind_cast<double>(rebind_cast<int64_t>(keepTopBit)) + 0.5;
    // return select((v & 0x8000000000000000) != 0, 0, min(64, 1086 - (simd_bit_cast<std::uint64_t>(fp) >> 52)));
  }
  else
    static_assert(dependent_false<_Tp>, "Unsupported data type for clz");
}

/// @brief Count the number of leading zero bits in each element
/// @param x The input value
/// @return A vec value in which each element contains the number of 1 bits in each respective input element.
/// @internal
#if defined(__AVX512F__)
template<typename _Tp, typename _Abi>
requires (sizeof(_Tp) == 4 || sizeof(_Tp) == 8) // Only works on 4 or 8 byte values
constexpr basic_vec<_Tp, _Abi> clz(x86_avx512_tag, const basic_vec<_Tp, _Abi>& x) noexcept {
  target_overloads impl {
    [=](xmm_register<std::uint32_t> auto v) {  return _mm_lzcnt_epi32(v.to_register()); },
    [=](xmm_register<std::uint64_t> auto v) {  return _mm_lzcnt_epi64(v.to_register()); },

    [=](ymm_register<std::uint32_t> auto v) {  return _mm256_lzcnt_epi32(v.to_register()); },
    [=](ymm_register<std::uint64_t> auto v) {  return _mm256_lzcnt_epi64(v.to_register()); },

    [=](zmm_register<std::uint32_t> auto v) {  return _mm512_lzcnt_epi32(v.to_register()); },
    [=](zmm_register<std::uint64_t> auto v) {  return _mm512_lzcnt_epi64(v.to_register()); },

    [=](auto unhandled) -> decltype(unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); }
  };

  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, x);
}
#endif

// :TODO: __AVX512VBMI2__ could handle byte and short by an epi8 double-register lookup.

/// @brief Reverse the order of the bits in each byte of the input.
/// @tparam _Np The number of elements in each vec
/// @param x The vec value whose bits will be reversed.
/// @return The bit-reversed vec value.
template<typename _Tp, typename _Abi>
requires (sizeof(_Tp) == 1) // Multi-byte reversal handled generically.
constexpr basic_vec<_Tp, _Abi> bitreverse(x86_sse_tag, const basic_vec<_Tp, _Abi>& x) noexcept {

  // Reverse the bits within individual bytes.
  const auto byteBitReverser = []<typename _Vec>(_Vec piece) -> _Vec {
    const auto bytes = simd_bit_cast<std::uint8_t>(piece);

    // Reverse upper and lower 4-bit blocks using a lut, and then combine them
    // together. Note that the lookup tables contain the data shifted into the
    // correct nibble, to avoid having to do that dynamically.
    constexpr std::uint8_t _revbytes[16] = {0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
    const auto lowRev  = shuffle_epi8_lut([](auto idx) -> std::uint8_t { return _revbytes[idx % 16] << 4; }, bytes & cw<0xF>);
    const auto highRev = shuffle_epi8_lut([](auto idx) -> std::uint8_t { return _revbytes[idx % 16]; }, bytes >> 4);

    return _Vec(lowRev | highRev);
  };

  return chunked_invoke(byteBitReverser, x);
}

/// @brief Reverse the order of the bits in each byte using the GFNI ISA.
/// @tparam _Np The number of elements in each vec
/// @param x The vec value whose bits will be reversed.
/// @return The bit-reverse vec value.
template<vec_integral _Vp>
requires (sizeof(typename _Vp::value_type) == 1) // Multi-byte reversal handled generically.
constexpr _Vp bitreverse(x86_avxsnc_tag, const _Vp& x) noexcept
  { return x86::gf2p8affine(x, uint64_t(0x8040201008040201)); }

template<typename _Tp, typename _Abi>
requires (detail::is_floating_point_v<_Tp>)
constexpr basic_vec<_Tp, _Abi> rcp(x86_sse_tag, const basic_vec<_Tp, _Abi>& v)
{
  target_overloads impl {
    [=](xmm_register<float> auto x)  { return _mm_rcp_ps(x.to_register());},

#if defined(__AVX__)
    [=](ymm_register<float> auto x)  { return _mm256_rcp_ps(x.to_register());},
#endif

    // Generic version for unhandled cases.
    [=](auto x) { return _Tp(1) / x; }
  };

  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, v);
}

#if defined(__AVX512F__)
template<std::floating_point _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> rcp(x86_avx512_tag, const basic_vec<_Tp, _Abi>& v)
{
  target_overloads impl {
    [=](xmm_register<float> auto x)  { return _mm_rcp14_ps(x.to_register());},
    [=](ymm_register<float> auto x)  { return _mm256_rcp14_ps(x.to_register());},

    [=](xmm_register<double> auto x) { return _mm_rcp14_pd(x.to_register());},
    [=](ymm_register<double> auto x) { return _mm256_rcp14_pd(x.to_register());},

    [=](zmm_register<float> auto x)  { return _mm512_rcp14_ps(x.to_register());},
    [=](zmm_register<double> auto x) { return _mm512_rcp14_pd(x.to_register());},
  };

  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, v);
}
#endif

#if defined(__AVX512FP16__)
template<typename _Abi>
constexpr basic_vec<_Float16, _Abi> rcp(x86_avxspr_tag, const basic_vec<_Float16, _Abi>& v)
{
  target_overloads impl {
    [=](xmm_register<_Float16> auto x)  { return _mm_rcp_ph(x.to_register());},
    [=](ymm_register<_Float16> auto x)  { return _mm256_rcp_ph(x.to_register());},
    [=](zmm_register<_Float16> auto x)  { return _mm512_rcp_ph(x.to_register());},
  };
  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, v);
}
#endif

template<typename _Tp, typename _Abi>
requires (detail::is_floating_point_v<_Tp>)
constexpr basic_vec<_Tp, _Abi> rsqrt(x86_sse_tag, const basic_vec<_Tp, _Abi>& v)
{
  target_overloads impl {
    [=](xmm_register<float> auto x)  { return _mm_rsqrt_ps(x.to_register());},
    [=](xmm_register<double> auto x) { return 1.0 / _mm_sqrt_pd(x.to_register());},

#if defined(__AVX__)
    [=](ymm_register<float> auto x)  { return _mm256_rsqrt_ps(x.to_register());},
    [=](ymm_register<double> auto x) { return 1.0 / _mm256_sqrt_pd(x.to_register());},
#endif

    // Generic version of all other cases.
    [=](auto x) { return _Tp(1) / sqrt(x); }
  };

  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, v);
}

#if defined(__AVX512F__)
template<std::floating_point _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> rsqrt(x86_avx512_tag, const basic_vec<_Tp, _Abi>& v)
{
  target_overloads impl {
    [=](xmm_register<float> auto x)  { return _mm_rsqrt14_ps(x.to_register());},
    [=](ymm_register<float> auto x)  { return _mm256_rsqrt14_ps(x.to_register());},
    
    [=](xmm_register<double> auto x) { return _mm_rsqrt14_pd(x.to_register());},
    [=](ymm_register<double> auto x) { return _mm256_rsqrt14_pd(x.to_register());},

    [=](zmm_register<float> auto x)  { return _mm512_rsqrt14_ps(x.to_register());},
    [=](zmm_register<double> auto x) { return _mm512_rsqrt14_pd(x.to_register());},
  };
  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, v);
}
#endif

#if defined(__AVX512FP16__)
template<typename _Abi>
constexpr basic_vec<_Float16, _Abi> rsqrt(x86_avxspr_tag, const basic_vec<_Float16, _Abi>& v)
{
  target_overloads impl {
    [=](xmm_register<_Float16> auto x)  { return _mm_rsqrt_ph(x.to_register());},
    [=](ymm_register<_Float16> auto x)  { return _mm256_rsqrt_ph(x.to_register());},
    [=](zmm_register<_Float16> auto x)  { return _mm512_rsqrt_ph(x.to_register());},
  };
  return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, v);
}
#endif

template<typename _Tp, typename _Abi>
requires (detail::is_floating_point_v<_Tp>)
constexpr basic_vec<_Tp, _Abi> sqrt(x86_sse_tag, const basic_vec<_Tp, _Abi>& v)
{
  target_overloads impl {
    [=](xmm_register<float> auto x)  { return _mm_sqrt_ps(x.to_register());},
    [=](xmm_register<double> auto x) { return _mm_sqrt_pd(x.to_register());},

#if defined(__AVX__)
    [=](ymm_register<float> auto x)  { return _mm256_sqrt_ps(x.to_register());},
    [=](ymm_register<double> auto x) { return _mm256_sqrt_pd(x.to_register());},
#endif

#if defined(__AVX512F__)
    [=](zmm_register<float> auto x)  { return _mm512_sqrt_ps(x.to_register());},
    [=](zmm_register<double> auto x) { return _mm512_sqrt_pd(x.to_register());},
#endif

#if defined(__AVX512FP16__)
    [=](xmm_register<_Float16> auto x) { return _mm_sqrt_ph(x.to_register());},
    [=](ymm_register<_Float16> auto x) { return _mm256_sqrt_ph(x.to_register());},
    [=](zmm_register<_Float16> auto x) { return _mm512_sqrt_ph(x.to_register());},
#endif

    [=](auto unhandled) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled; }
  };

  if constexpr (is_fp16_v<_Tp> && !hasAvxFp16)
  {
    // When sqrt is needed in FP16 use FP32 to do the work.
    const auto sqrt32 = sqrt(rebind_cast<float>(v));
    return rebind_cast<_Tp>(sqrt32);
  }
  else
    return chunked_invoke([=](auto v) { return decltype(v)(impl(v)); }, v);
}

/// @brief Compute the saturated addition of each respective pair of SIMD elements
/// @ingroup simd_numeric
/// @tparam _Tp The integral type of each basic_vec element.
/// @tparam _Abi The ABI of the basic_vec
/// @param v The basic_vec value
/// @return A basic_vec value in which each element is the saturating addition of the respective input basic_vec elements.
template<std::integral _Tp, typename _Abi>
requires (sizeof(_Tp) <= 2) // x86 only has support for 8 and 16-bit saturating operations. Defer everything else to the
                            // generic synthesised version.
constexpr basic_vec<_Tp, _Abi> saturating_add(x86_tag, const basic_vec<_Tp, _Abi>& lhs, const basic_vec<_Tp, _Abi>& rhs) noexcept
{
  target_overloads impl {
    // 128-bit
    [=](xmm_register<uint8_t> auto x, auto y) { return _mm_adds_epu8(x.to_register(), y.to_register()); },
    [=](xmm_register<int8_t> auto x, auto y) { return _mm_adds_epi8(x.to_register(), y.to_register()); },
    [=](xmm_register<uint16_t> auto x, auto y) { return _mm_adds_epu16(x.to_register(), y.to_register()); },
    [=](xmm_register<int16_t> auto x, auto y) { return _mm_adds_epi16(x.to_register(), y.to_register()); },

    // 256-bit (but only in Intel AVX2 - Intel AVX has to break into 128-bit pieces instead).
#if defined(__AVX2__)
    [=](ymm_register<uint8_t> auto x, auto y) { return _mm256_adds_epu8(x.to_register(), y.to_register()); },
    [=](ymm_register<int8_t> auto x, auto y) { return _mm256_adds_epi8(x.to_register(), y.to_register()); },
    [=](ymm_register<uint16_t> auto x, auto y) { return _mm256_adds_epu16(x.to_register(), y.to_register()); },
    [=](ymm_register<int16_t> auto x, auto y) { return _mm256_adds_epi16(x.to_register(), y.to_register()); },
#endif

    // 512-bit
#if defined(__AVX512F__)
    [=](zmm_register<uint8_t> auto x, auto y) { return _mm512_adds_epu8(x.to_register(), y.to_register()); },
    [=](zmm_register<int8_t> auto x, auto y) { return _mm512_adds_epi8(x.to_register(), y.to_register()); },
    [=](zmm_register<uint16_t> auto x, auto y) { return _mm512_adds_epu16(x.to_register(), y.to_register()); },
    [=](zmm_register<int16_t> auto x, auto y) { return _mm512_adds_epi16(x.to_register(), y.to_register()); },
 #endif

    [=](auto unhandled, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled;}
  };

  // 256-bit saturating only exists in the Intel AVX2 ISA, not Intel AVX. If
  // Intel AVX2 is not available break the problem into 128-bit chunks instead
  // to use Intel SSE ISA.
  constexpr int numElements = (hasAvx && !hasAvx2) ? vec<_Tp>::size / 2 : vec<_Tp>::size;
  auto wrapper = [=](auto a, auto b) { return decltype(a)(impl(a, b)); }; // Return a vec from an intrinsic.
  return chunked_invoke<numElements>(wrapper, lhs, rhs);
}

/// @brief Compute the saturated subtraction of each respective pair of SIMD elements
/// @ingroup simd_numeric
/// @tparam _Tp The integral type of each basic_vec element.
/// @tparam _Abi The ABI of the basic_vec
/// @param v The basic_vec value
/// @return A basic_vec value in which each element is the saturating subtraction of the respective input basic_vec elements.
template<std::integral _Tp, typename _Abi>
requires (sizeof(_Tp) <= 2) // x86 only has support for 8 and 16-bit saturating operations. Defer everything else to the
                            // generic synthesised version.
constexpr basic_vec<_Tp, _Abi> saturating_sub(x86_tag, const basic_vec<_Tp, _Abi>& lhs, const basic_vec<_Tp, _Abi>& rhs) noexcept
{
  target_overloads impl {
    // 128-bit
    [=](xmm_register<uint8_t> auto x, auto y) { return _mm_subs_epu8(x.to_register(), y.to_register()); },
    [=](xmm_register<int8_t> auto x, auto y) { return _mm_subs_epi8(x.to_register(), y.to_register()); },
    [=](xmm_register<uint16_t> auto x, auto y) { return _mm_subs_epu16(x.to_register(), y.to_register()); },
    [=](xmm_register<int16_t> auto x, auto y) { return _mm_subs_epi16(x.to_register(), y.to_register()); },

    // 256-bit (but only in Intel AVX2 - Intel AVX has to break into 128-bit pieces instead).
#if defined(__AVX2__)
    [=](ymm_register<uint8_t> auto x, auto y) { return _mm256_subs_epu8(x.to_register(), y.to_register()); },
    [=](ymm_register<int8_t> auto x, auto y) { return _mm256_subs_epi8(x.to_register(), y.to_register()); },
    [=](ymm_register<uint16_t> auto x, auto y) { return _mm256_subs_epu16(x.to_register(), y.to_register()); },
    [=](ymm_register<int16_t> auto x, auto y) { return _mm256_subs_epi16(x.to_register(), y.to_register()); },
#endif

    // 512-bit
#if defined(__AVX512F__)
    [=](zmm_register<uint8_t> auto x, auto y) { return _mm512_subs_epu8(x.to_register(), y.to_register()); },
    [=](zmm_register<int8_t> auto x, auto y) { return _mm512_subs_epi8(x.to_register(), y.to_register()); },
    [=](zmm_register<uint16_t> auto x, auto y) { return _mm512_subs_epu16(x.to_register(), y.to_register()); },
    [=](zmm_register<int16_t> auto x, auto y) { return _mm512_subs_epi16(x.to_register(), y.to_register()); },
 #endif

    [=](auto unhandled, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled;}
  };

  // 256-bit saturating only exists in the Intel AVX2 ISA, not Intel AVX. If
  // Intel AVX2 is not available break the problem into 128-bit chunks instead
  // to use Intel SSE ISA.
  constexpr int numElements = (hasAvx && !hasAvx2) ? vec<_Tp>::size / 2 : vec<_Tp>::size;
  auto wrapper = [=](auto a, auto b) { return decltype(a)(impl(a, b)); }; // Return a vec from an intrinsic.
  return chunked_invoke<numElements>(wrapper, lhs, rhs);
}

#if defined(__AVX512VBMI2__)
template<vec_unsigned_integral _Vp, vec_integral _Sp>
  requires (sizeof(_Vp::value_type) >= 2) // __AVX512VBMI2__ only has variable shift intrinsics for 16, 32, and 64-bit integers.
constexpr _Vp fsr(x86_avxsnc_tag, const _Vp& high, const _Vp& low, const _Sp& shift)
{
  using namespace detail;

  target_overloads impl {

    [=]<xmm_register<std::uint16_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm_shrdv_epi16(x.to_register(), y.to_register(), s.to_register()); },
    [=]<ymm_register<std::uint16_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm256_shrdv_epi16(x.to_register(), y.to_register(), s.to_register()); },
    [=]<zmm_register<std::uint16_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm512_shrdv_epi16(x.to_register(), y.to_register(), s.to_register()); },

    [=]<xmm_register<std::uint32_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm_shrdv_epi32(x.to_register(), y.to_register(), s.to_register()); },
    [=]<ymm_register<std::uint32_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm256_shrdv_epi32(x.to_register(), y.to_register(), s.to_register()); },
    [=]<zmm_register<std::uint32_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm512_shrdv_epi32(x.to_register(), y.to_register(), s.to_register()); },

    [=]<xmm_register<std::uint64_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm_shrdv_epi64(x.to_register(), y.to_register(), s.to_register()); },
    [=]<ymm_register<std::uint64_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm256_shrdv_epi64(x.to_register(), y.to_register(), s.to_register()); },
    [=]<zmm_register<std::uint64_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm512_shrdv_epi64(x.to_register(), y.to_register(), s.to_register()); },

    [=](auto unhandled, auto, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled; }
  };

  auto wrapper = [=]<typename _Vec>(_Vec x, _Vec y, _Vec s) { return _Vec(impl(x, y, s)); };
  return std::bit_cast<_Vp>(chunked_invoke(wrapper, low, high, shift));
}

template<vec_unsigned_integral _Vp, vec_integral _Sp>
  requires (sizeof(_Vp::value_type) >= 2) // __AVX512VBMI2__ only has variable shift intrinsics for 16, 32, and 64-bit integers.
constexpr _Vp fsl(x86_avxsnc_tag, const _Vp& high, const _Vp& low, const _Sp& shift)
{
  using namespace detail;

  target_overloads impl {
    [=]<xmm_register<std::uint16_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm_shldv_epi16(x.to_register(), y.to_register(), s.to_register()); },
    [=]<ymm_register<std::uint16_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm256_shldv_epi16(x.to_register(), y.to_register(), s.to_register()); },
    [=]<zmm_register<std::uint16_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm512_shldv_epi16(x.to_register(), y.to_register(), s.to_register()); },

    [=]<xmm_register<std::uint32_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm_shldv_epi32(x.to_register(), y.to_register(), s.to_register()); },
    [=]<ymm_register<std::uint32_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm256_shldv_epi32(x.to_register(), y.to_register(), s.to_register()); },
    [=]<zmm_register<std::uint32_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm512_shldv_epi32(x.to_register(), y.to_register(), s.to_register()); },

    [=]<xmm_register<std::uint64_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm_shldv_epi64(x.to_register(), y.to_register(), s.to_register()); },
    [=]<ymm_register<std::uint64_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm256_shldv_epi64(x.to_register(), y.to_register(), s.to_register()); },
    [=]<zmm_register<std::uint64_t> _Vec>(_Vec x, _Vec y, _Vec s)
      { return _mm512_shldv_epi64(x.to_register(), y.to_register(), s.to_register()); },
 
    [=](auto unhandled, auto, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled; }
  };

  auto wrapper = [=]<typename _Vec>(_Vec x, _Vec y, _Vec s) { return _Vec(impl(x, y, s)); };
  return std::bit_cast<_Vp>(chunked_invoke(wrapper, low, high, shift));
}
#endif // __AVX512VBMI2__

} // End of detail namespace

} // namespace _XVEC_NAMESPACE::simd
