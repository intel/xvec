//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#define _XVEC_UNARY_MATHS_OP(NAME) \
template<_XVEC_NAMESPACE::simd::detail::math_floating_point _Vp> \
constexpr _XVEC_NAMESPACE::simd::detail::deduced_vec_t<_Vp> NAME (const _Vp& v) { \
constexpr auto op = [](auto v) { using std::NAME ; return NAME(v); }; \
return _XVEC_NAMESPACE::simd::detail::maths_fn(v, op); }

#define _XVEC_BINARY_MATHS_OP(NAME) \
template<_XVEC_NAMESPACE::simd::detail::math_floating_point _Vp> \
constexpr auto NAME (const _Vp& v0, const _Vp& v1) { \
constexpr auto op = [](auto lhs, auto rhs) { using std::NAME ; return NAME(lhs, rhs); }; \
return _XVEC_NAMESPACE::simd::detail::maths_fn(v0, v1, op); }

namespace _XVEC_NAMESPACE::simd::detail {

template<math_floating_point _Vp>
constexpr deduced_vec_t<_Vp> maths_fn(const _Vp& v, auto op)
{
    using _Tp = typename _Vp::value_type;

    if constexpr (_XVEC_NAMESPACE::simd::detail::is_fp16_v<_Tp>)
      // Emulating FP16 on a non-native machine uses float to do the work.
      return rebind_cast<_Tp>(maths_fn(generic_tag{}, rebind_cast<float>(v), op));
    else
      return deduced_vec_t<_Vp>([=](auto i) { return op(v[i]); });
}

template<math_floating_point _Vp>
constexpr deduced_vec_t<_Vp> maths_fn(const _Vp& v0, const _Vp& v1, auto op)
{
    using _Tp = typename _Vp::value_type;
    if constexpr (_XVEC_NAMESPACE::simd::detail::is_fp16_v<_Tp>)
      // Emulating FP16 on a non-native machine uses float to do the work.
      return rebind_cast<_Tp>(maths_fn(generic_tag{}, rebind_cast<float>(v0), rebind_cast<float>(v1), op));
    else
      return deduced_vec_t<_Vp>([=](auto i) { return op(v0[i], v1[i]); });
}


} // namespace _XVEC_NAMESPACE::simd::detail
