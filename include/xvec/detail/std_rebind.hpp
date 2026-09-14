//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <complex>
#include <utility>
#include <cstddef>

#include <xvec/detail/config.hpp>

// NOTE: rebind_cast deliberately lives one level above `simd`, because P3971R1
// places it in namespace std. It is a general element-type rebinding facility
// spanning sequence containers, std::complex and user-defined uniform-element
// types, of which simd is only one participant; only that participant is
// implemented here. The simd-specific `simd::rebind_t` alias from C++26 is a
// separate entity (see detail/core.hpp); the two agree on simd types.
namespace _XVEC_NAMESPACE::detail
{
/// \defgroup rebind facilities.
/// @brief rebind_cast implementation, as per P3971R1.

/// Rebind simd vector.
template<typename _Up, _XVEC_NAMESPACE::simd::vec_type _Vp>
constexpr _XVEC_NAMESPACE::simd::rebind_t<_Up, _Vp> rebind_cast(const _Vp& v)
  { return _XVEC_NAMESPACE::simd::rebind_t<_Up, _Vp>(v); }

/// Customisation point object for rebind_cast.
template<typename _Up>
struct RebindCast {
  template<typename _Tp>
  constexpr auto operator()(_Tp&& t) const -> decltype(rebind_cast<_Up>(std::forward<_Tp>(t))) {
      return rebind_cast<_Up>(std::forward<_Tp>(t));
  }
};

} // namespace _XVEC_NAMESPACE::detail

namespace _XVEC_NAMESPACE
{
  /// Rebinding CPO.
  template<typename _Up> inline constexpr detail::RebindCast<_Up> rebind_cast{};
}
