//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <immintrin.h>

#include <xvec/detail/config.hpp>

#include <concepts>

namespace _XVEC_NAMESPACE::simd
{

/// A generic tag is used for the default implementation of raw simd operations.
/// Other vendors and target-specific tags can be provided to override the
/// generic operations.
struct generic_tag {};

/// If a type supplied to get_register_type is invalid, mark that as such with
/// this special type.
struct invalid_register {};

#if defined(__SSE__) and !defined(_XVEC_FORCE_SCALAR)

// These target tags are specific to x86 and provide a hierarchy of
// capabilities. Note that one of the tags must be compact_mask_tag, which
// indicates that the target supports compact masks. This is used to select the
// default implementation of some operations which can be more efficient if
// compact masks are supported. The tag must appear unambiguously in the
// hierarchy so that the tag overloads always resolve to exactly one option
// (e.g., no multiple inheritance to make several tags qualify).
struct x86_tag : public generic_tag {};
struct x86_sse_tag : public x86_tag {};
struct x86_avx_tag : public x86_sse_tag {};
struct x86_avx2_tag : public x86_avx_tag {};
struct compact_mask_tag : public x86_avx2_tag {};
struct x86_avx512_tag : public compact_mask_tag {};
struct x86_avxsnc_tag : public x86_avx512_tag {};
struct x86_avxspr_tag : public x86_avxsnc_tag {};
struct x86_avx10_tag : public x86_avxspr_tag {};

#if defined(__AVX10_2__)
  using target_tag = x86_avx10_tag;
#elif defined(__AVX512FP16__)
  using target_tag = x86_avxspr_tag;
#elif defined(__AVX512VPOPCNTDQ__)
  using target_tag = x86_avxsnc_tag;
#elif defined(__AVX512F__)
  using target_tag = x86_avx512_tag;
#elif defined (__AVX2__)
  using target_tag = x86_avx2_tag;
#elif defined (__AVX__)
  using target_tag = x86_avx_tag;
#else
  using target_tag = x86_sse_tag;
#endif

namespace detail
{

/// Compile time flag to provide short-hand to query a few common ISA variants.
inline constexpr bool hasSse = std::derived_from<target_tag, x86_sse_tag>;
inline constexpr bool hasAvx = std::derived_from<target_tag, x86_avx_tag>;
inline constexpr bool hasAvx2 = std::derived_from<target_tag, x86_avx2_tag>;
inline constexpr bool hasAvx512 = std::derived_from<target_tag, x86_avx512_tag>;
inline constexpr bool hasAvxFp16 = std::derived_from<target_tag, x86_avxspr_tag>;

/// Compile time constant indicating how many bytes are available in the largest register on the current target.
inline constexpr int maxBytesInVec = hasAvx512 ? 64 : hasAvx ? 32 : hasSse ? 16 : 0;

/// Return how many elements to use in a vec in order to make best use
/// of the registers. For small vec values it will choose the smallest register
/// size that stores the given number of elements. For vec values bigger than
/// the register size it will return the largest available register
template<typename _Tp, int _Np>
inline constexpr int register_size =
    (hasAvx512 && (sizeof(_Tp) * _Np) > 32) ? (64 / sizeof(_Tp)) :
    (hasAvx    && (sizeof(_Tp) * _Np) > 16) ? (32 / sizeof(_Tp)) :
    16 / sizeof(_Tp);

/// Generate the appropriate register type for the given vec type. For example,
/// a small vector of ints would be converted to the smallest native register
/// capable of storing all the bits, with padding added to bring it up to the
/// register size. the type of the element is also taken into account (e.g.,
/// `double` elements would convert to a `d` suffix like __m256d).
template<typename T>
consteval auto get_register_type() {
    constexpr auto bytes = sizeof(T);

    using _Tp = std::remove_cvref_t<decltype(std::declval<T&>()[0])>;

    constexpr bool isFloat = std::same_as<float, _Tp>;
    constexpr bool isDouble = std::same_as<double, _Tp>;

    if constexpr (bytes > maxBytesInVec)
      return invalid_register{};
    else
#if defined(__AVX512FP16__)
    if constexpr (std::same_as<_Tp, _Float16>)
    {
        if constexpr      (bytes > 32) return __m512h();
        else if constexpr (bytes > 16) return __m256h();
        else                           return __m128h();
    } else
#endif
#if defined(__AVX512F__)
    if constexpr (bytes > 32)
    {
        if constexpr      (isFloat)  return __m512();
        else if constexpr (isDouble) return __m512d();
        else                         return __m512i();
    } else
#endif
#if defined(__AVX__)
    if constexpr (bytes > 16)
    {
        if constexpr      (isFloat)  return __m256();
        else if constexpr (isDouble) return __m256d();
        else                         return __m256i();
    } else
#endif
    {
        if constexpr      (isFloat)  return __m128();
        else if constexpr (isDouble) return __m128d();
        else                         return __m128i();
    }
}

} // namespace detail

#else

namespace detail
{
/// Generic version must make some assumption about the maximum register size
/// and mask type in order to provide some basic functionality. These can be
/// overridden by targets which have different capabilities. In practice, a
/// completely generic version would be scalar, but the default is set to
/// 128-bit vectors and wide masks to allow the library to be used in a more
/// efficient way on targets which support that without needing to write
/// target-specific code.
inline constexpr int maxBytesInVec = 16;
template<typename _Tp, int _Np> inline constexpr int register_size = maxBytesInVec / sizeof(_Tp);

/// In generic targets, there is no way to know what the register type should
/// be, so return an invalid_register type to indicate that the target does not
/// support the requested type. This will cause a compile-time error if the user
/// tries to use a type which is not supported by the target. Targets which
/// support a specific register type should provide a specialization of this
/// function which returns the appropriate register type.
template<typename> consteval auto get_register_type() { return invalid_register{}; }

}

struct compact_mask_tag : public generic_tag {};

#if defined(_XVEC_GENERIC_COMPACT)
  using target_tag = compact_mask_tag;
#else
  using target_tag = generic_tag;
#endif

#endif

/// Detect the broad category of compact mask support. This is used to query the
/// capability of the target to support compact masks, which can be more
/// efficient for some operations. The tag must be present in the target tag
/// hierarchy.
template<typename Tag> concept has_compact_mask = std::derived_from<Tag, compact_mask_tag>;

inline constexpr auto target = target_tag{};

} // namespace _XVEC_NAMESPACE::simd
