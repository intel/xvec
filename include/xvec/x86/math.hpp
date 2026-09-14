//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

/// Sadly, macros are the easiest way to do things here. The boilerplate code
/// for tens of functions which are simply invoking a named intrinsic is huge,
/// so it makes things much easier to wrap all the code inside a macro which
/// cookie cuts the right structure with minimal effort.

#define _XVEC_SSE_UNARY_MATHS_OP(NAME) [](__m128 v) { return _mm_ ##NAME ##_ps (v); }, [](__m128d v) { return _mm_ ##NAME ##_pd (v); },
#define _XVEC_SSE_BINARY_MATHS_OP(NAME) [](__m128 v0, __m128 v1) { return _mm_ ##NAME ##_ps (v0, v1); }, [](__m128d v0, __m128d v1) { return _mm_ ##NAME ##_pd (v0, v1); },

#if defined (__AVX__)
#define _XVEC_AVX_UNARY_MATHS_OP(NAME) [](__m256 v) { return _mm256_ ##NAME ##_ps (v); }, [](__m256d v) { return _mm256_ ##NAME ##_pd (v); },
#define _XVEC_AVX_BINARY_MATHS_OP(NAME) [](__m256 v0, __m256 v1) { return _mm256_ ##NAME ##_ps (v0, v1); }, [](__m256d v0, __m256d v1) { return _mm256_ ##NAME ##_pd (v0, v1); },
#else
#define _XVEC_AVX_UNARY_MATHS_OP(NAME)
#define _XVEC_AVX_BINARY_MATHS_OP(NAME)
#endif

#if defined (__AVX512F__)
#define _XVEC_AVX512_UNARY_MATHS_OP(NAME) [](__m512 v) { return _mm512_ ##NAME ##_ps (v); }, [](__m512d v) { return _mm512_ ##NAME ##_pd (v); },
#define _XVEC_AVX512_BINARY_MATHS_OP(NAME) [](__m512 v0, __m512 v1) { return _mm512_ ##NAME ##_ps (v0, v1); }, [](__m512d v0, __m512d v1) { return _mm512_ ##NAME ##_pd (v0, v1); },
#else
#define _XVEC_AVX512_UNARY_MATHS_OP(NAME)
#define _XVEC_AVX512_BINARY_MATHS_OP(NAME)
#endif

#if defined (__AVX512FP16__)
  #define _XVEC_AVX512FP16_UNARY_MATHS_OP(NAME) [](__m128h v) { return _mm_ ##NAME ##_ph (v); }, [](__m256h v) { return _mm256_ ##NAME ##_ph (v); }, [](__m512h v) { return _mm512_ ##NAME ##_ph (v); },
  #define _XVEC_AVX512FP16_BINARY_MATHS_OP(NAME) [](__m128h v0, __m128h v1) { return _mm_ ##NAME ##_ph (v0, v1); }, [](__m256h v0, __m256h v1) { return _mm256_ ##NAME ##_ph (v0, v1); }, [](__m512h v0, __m512h v1) { return _mm512_ ##NAME ##_ph (v0, v1); },
#else
#define _XVEC_AVX512FP16_UNARY_MATHS_OP(NAME)
#define _XVEC_AVX512FP16_BINARY_MATHS_OP(NAME)
#endif

// The following creates a lambda inside detail to provide overloads for
// all the interesting intrinsics, and then sets up the actual xvec interface
// function to invoke it.
#define _XVEC_UNARY_MATHS_OP(NAME) \
  template<_XVEC_NAMESPACE::simd::detail::math_floating_point _Vp> \
  constexpr _XVEC_NAMESPACE::simd::detail::deduced_vec_t<_Vp> NAME (const _Vp& v) { \
    _XVEC_NAMESPACE::simd::detail::target_overloads op { \
      _XVEC_SSE_UNARY_MATHS_OP(NAME)         \
      _XVEC_AVX_UNARY_MATHS_OP(NAME)         \
      _XVEC_AVX512_UNARY_MATHS_OP(NAME)      \
      _XVEC_AVX512FP16_UNARY_MATHS_OP(NAME)  \
      [](auto unhandled) { static_assert(_XVEC_NAMESPACE::simd::detail::dependent_false<decltype(unhandled)>); return unhandled;} \
    }; \
    return _XVEC_NAMESPACE::simd::detail::maths_fn(v, op); \
  }

#define _XVEC_BINARY_MATHS_OP(NAME) \
  template<_XVEC_NAMESPACE::simd::detail::math_floating_point _Vp> \
  constexpr auto NAME (const _Vp& v0, const _Vp& v1) { \
    _XVEC_NAMESPACE::simd::detail::target_overloads op { \
      _XVEC_SSE_BINARY_MATHS_OP(NAME)         \
      _XVEC_AVX_BINARY_MATHS_OP(NAME)         \
      _XVEC_AVX512_BINARY_MATHS_OP(NAME)      \
      _XVEC_AVX512FP16_BINARY_MATHS_OP(NAME)  \
      [](auto unhandled, auto) { static_assert(_XVEC_NAMESPACE::simd::detail::dependent_false<decltype(unhandled)>); return unhandled;} \
    }; \
    return _XVEC_NAMESPACE::simd::detail::maths_fn(v0, v1, op); \
  }

namespace _XVEC_NAMESPACE::simd::detail
{
/// Invoke an overloaded intrinsic data set representing a common maths operation onto the supplied SIMD value.
/// @tparam _Tp The type of the SIMD elements.
/// @tparam _Abi The ABI of the SIMD
/// @tparam OP_IMPL the overload set to invoke
/// @param v The SIMD value used as input.
/// @param op_impl The operation to invoke.
/// @return The result of applying the overloaded intrinsic to all of the
/// elements of the SIMD value. Integral values will be promoted to double
/// precision.
template<math_floating_point _Vp, typename OP_IMPL>
constexpr deduced_vec_t<_Vp> maths_fn(const _Vp& v, OP_IMPL op_impl)
{
  using _Tp = typename _Vp::value_type;
  if constexpr (detail::is_fp16_v<_Tp> && !detail::hasAvxFp16)
    // Emulating FP16 on a non-native machine uses float to do the work.
    return rebind_cast<_Tp>(maths_fn(rebind_cast<float>(v), op_impl));
  else
  {
    // All other floating point types (including native FP16) get dispatched to the
    // appropriate intrinsic.
    auto wrapper = [=]<typename _Vec>(_Vec x) { return _Vec(op_impl(x.to_register())); };
    return chunked_invoke(wrapper, v);
  }
}

template<math_floating_point _Vp, typename OP_IMPL>
constexpr deduced_vec_t<_Vp> maths_fn(const _Vp& v0, const _Vp& v1, OP_IMPL op_impl)
{
  using _Tp = typename _Vp::value_type;
  if constexpr (_XVEC_NAMESPACE::simd::detail::is_fp16_v<_Tp> && !_XVEC_NAMESPACE::simd::detail::hasAvxFp16)
    // Emulating FP16 on a non-native machine uses float to do the work.
    return rebind_cast<_Tp>(maths_fn(rebind_cast<float>(v0), rebind_cast<float>(v1), op_impl));
  else
  {
    // All other floating point types (including native FP16) get dispatched to the
    // appropriate intrinsic.
    auto wrapper = [=]<typename _Vec>(_Vec x, _Vec y) { return _Vec(op_impl(x.to_register(), y.to_register())); };
    return chunked_invoke(wrapper, v0, v1);
  }
}

} // namespace _XVEC_NAMESPACE::simd::detail
