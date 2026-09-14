//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <bitset>
#include <cstddef> // std::size_t
#include <type_traits>
#include <algorithm> // std::min, std::max
#include <utility> // std::index_sequence
#include <cstring> // for std::memcpy

#include <xvec/detail/utilities.hpp>
#include <xvec/detail/core.hpp>

namespace _XVEC_NAMESPACE::simd::detail
{

/// General purpose unary operator. Passes through to the compiler's own builtin.
template<vec_or_mask_type _Vp, typename _Op>
constexpr auto builtin_operator(generic_tag, _Vp value, _Op op) {
  auto t = value.to_builtin();
  decltype(op(t)) r;

  // Ensure that the target ISA for op() is embedded here.
#if __clang_major__ > 14
  // Clang 15+ uses a different attribute to force inlining.
  [[clang::always_inline]] r = op(t);
#else
  #pragma forceinline
  r = op(t);
#endif

  if constexpr (mask_operator<_Op>) return _Vp::mask_type::from_builtin(r);
  else return _Vp::from_builtin(r);
}

/// Provide a default for !x. This removes the need for compact-mask operations
/// to have to do anything special.
template<vec_type _Vp>
constexpr auto builtin_operator(generic_tag, _Vp value, std::logical_not<>) { return value == _Vp(); }

/// General purpose binary operator.
template<vec_or_mask_type _Vp, typename _Op> // Concept - binary operator of some type
constexpr auto builtin_operator(generic_tag, _Vp lhs, _Vp rhs, _Op op) {
  auto lhsd = lhs.to_builtin();
  auto rhsd = rhs.to_builtin();
  decltype(op(lhsd, rhsd)) r;
  
  // Ensure that the target ISA for op() is embedded here.
#if __clang_major__ > 14
  // Clang 15+ uses a different attribute to force inlining.
  [[clang::always_inline]] r = op(lhsd, rhsd);
#else
  #pragma forceinline
  r = op(lhsd, rhsd);
#endif

  if constexpr (mask_operator<_Op>) return _Vp::mask_type::from_builtin(r);
  else return _Vp::from_builtin(r);
}

/// @brief Special treatment for integral divide and modulus. These often cause hardware
/// exceptions if zero is used in the denominator. Such exceptions should only
/// be raised on the active data in the vec.
template<vec_integral _Vp, typename _Op>
requires (any_same_v<_Op, std::divides<>, std::modulus<>>)
constexpr _Vp builtin_operator(generic_tag, _Vp lhs, _Vp rhs, _Op)
  { return _Vp([=](auto i) -> typename _Vp::value_type { return _Op{}(lhs[i], rhs[i]); }); }

/// Generic complex multipler forwards to the complex FMA.
template<vec_complex _Vp>
constexpr auto builtin_operator(generic_tag, const _Vp& lhs, const _Vp& rhs, std::multiplies<>)
  { return fma(lhs, rhs, _Vp{});}

/// Generic comparator for compact mask targets. This is almost certainly
/// inefficient so the target should provide something better.
template<vec_type _Vp, mask_operator _Op>
constexpr auto builtin_operator(compact_mask_tag, const _Vp& x, const _Vp& y, _Op)
  { return typename _Vp::mask_type{[=](auto i) -> bool { return _Op{}(x[i], y[i]); }}; }

/// Generic complex equality for wide masks. Each `bit' of the mask result will
/// be a container which is the same size as _Tp. Adjacent values are compared for
/// equality or inequality.
template<vec_complex _Vp, mask_operator _Op>
constexpr typename _Vp::mask_type builtin_operator(generic_tag, _Vp lhs, _Vp rhs, _Op) {
  using _Tp = typename _Vp::value_type::value_type;

  static_assert(sizeof(_Tp) <= 8, "Can't deal with 128-bit elements like complex<double> yet");

  using binop_type = std::remove_cvref_t<_Op>;
  constexpr bool isEqualOp = std::is_same_v<binop_type, std::equal_to<>>;
  constexpr bool isNotEqualOp = std::is_same_v<binop_type, std::not_equal_to<>>;
  static_assert(isEqualOp || isNotEqualOp, "Complex values only allow equality comparisons");

  // Element-wise comparison to generate a mask for each individual value.
  const auto compareElements = _Op{}(simd_bit_cast<_Tp>(lhs), simd_bit_cast<_Tp>(rhs));

  // Treat the mask as having half as many elements which are twice as big. Adjacent elements will be 0 or -1.
  const auto cmplxMask = simd_bit_cast<container_for_type<typename _Vp::value_type>>(-compareElements);

  // Compare wider elements to all 0 or all 1.
  if constexpr (isEqualOp)
    return typename _Vp::mask_type(cmplxMask == cw<-1>);
  else
    return typename _Vp::mask_type(cmplxMask != cw<0>);
}

/// Generate mask from binary operator. Converts operator result to boolean,
/// then to mask element. Generates wide mask first, then converts to
/// basic_mask - this two-step approach enables better compiler auto-vectorisation
/// than direct element-wise generation of compact masks.
template<typename FN, vec_type _Vp, typename... Args>
  requires (std::same_as<Args, _Vp> && ...)
constexpr auto make_mask_from_op(const _Vp& first, const Args&... args) noexcept {
  using _UnsignedMask = typename _Vp::mask_type::traits::unsigned_vec_for_mask;
  using _ME = typename _UnsignedMask::value_type;
  auto m = _UnsignedMask{[=](auto i) -> _ME { 
    return FN{}(first[i], (args[i])...) ? ~_ME() : _ME(); 
  }};
  return typename _Vp::mask_type(m != _UnsignedMask());
}

/// Generic operator dispatcher. This is the main entry point for all operators.
/// It dispatches to the appropriate implementation based on the type of the
/// operator, the input, return type, whether there is a customisation point,
/// whether it is a user-defined type, and so on.
template<typename FN, vec_type _Vp, vec_type... _Vps>
constexpr auto do_operator(const _Vp& first, const _Vps&... rest) noexcept {
  // Functions which return bool will return masks, otherwise return the same type as the input.
  using _Rp = std::conditional_t<mask_operator<FN>, typename _Vp::mask_type, _Vp>;

  if (std::is_constant_evaluated())
    // Simplest to call the generic constructor. No need for target-specific.
    return _Rp{[=](auto i) { return typename _Rp::value_type(FN{}(first[i], rest[i]...)); }}; 
  else if constexpr (builtin_vectorizable<typename _Vp::value_type>)
    // Builtin types can call the compiler's support directly.
    return detail::builtin_operator(target, first, rest..., FN{});
  else if constexpr (requires { {simd_operator(first, rest..., FN{})}; })
    // Custom optimised function. The builtin above takes precedence to avoid
    // the user overriding a target's own implementation.
    return simd_operator(first, rest..., FN{});
  else if constexpr (mask_operator<FN>)
    // Auto-vectorise a mask generator. For performance, mask operators go
    // through several steps which allow the compiler to auto-vectorise
    // effectively.
    return make_mask_from_op<FN>(first, rest...);
  else
    // Auto-vectorise a normal operator. This is the fallback for non-builtin
    // types which don't have a custom implementation.
    return _Rp{[=](auto i) { return static_cast<typename _Vp::value_type>(FN{}(first[i], rest[i]...)); }};
}

/// Conversion operator for basic_vec. All gate-keeping has been done, just do the convert.
template<vec_type _To, vec_type _From>
constexpr _To convert_operator(const _From& v) noexcept {
  using _Tp = typename _To::value_type;
  if constexpr (detail::builtin_vectorizable<_Tp> && detail::builtin_vectorizable<typename _From::value_type>)
    return _To::from_builtin(__builtin_convertvector(v.to_builtin(), typename _To::builtin_type));
  else if constexpr (requires {
    { simd_convert(v, _XVEC_NAMESPACE::simd::convert_to<_Tp>) } -> std::convertible_to<_To>;
  })
    return simd_convert(v, _XVEC_NAMESPACE::simd::convert_to<_Tp>);
  else
    return _To([=](auto i) { return static_cast<_Tp>(v[i]); });
}

} // namespace _XVEC_NAMESPACE::simd::detail
