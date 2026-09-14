//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <concepts>
#include <type_traits>
#include <limits>

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>

namespace std
{

using _XVEC_NAMESPACE::simd::basic_vec;

template <typename _Tp, typename _Abi>
  requires numeric_limits<_Tp>::is_specialized
struct numeric_limits<basic_vec<_Tp, _Abi>> {
private:
    using V = basic_vec<_Tp, _Abi>;
    using NL = numeric_limits<_Tp>;

public:

    static constexpr bool is_specialized = true;

    // Value-returning members: broadcast the scalar limit.
    static constexpr V min()           noexcept { return V(NL::min()); }
    static constexpr V max()           noexcept { return V(NL::max()); }
    static constexpr V lowest()        noexcept { return V(NL::lowest()); }
    static constexpr V epsilon()       noexcept { return V(NL::epsilon()); }
    static constexpr V round_error()   noexcept { return V(NL::round_error()); }
    static constexpr V infinity()      noexcept { return V(NL::infinity()); }
    static constexpr V quiet_NaN()     noexcept { return V(NL::quiet_NaN()); }
    static constexpr V signaling_NaN() noexcept { return V(NL::signaling_NaN()); }
    static constexpr V denorm_min()    noexcept { return V(NL::denorm_min()); }

    // Trait members: forward unchanged from numeric_limits<T>.
    static constexpr int  digits         = NL::digits;
    static constexpr int  digits10       = NL::digits10;
    static constexpr int  max_digits10   = NL::max_digits10;
    static constexpr bool is_signed      = NL::is_signed;
    static constexpr bool is_integer     = NL::is_integer;
    static constexpr bool is_exact       = NL::is_exact;
    static constexpr int  radix          = NL::radix;
    static constexpr int  min_exponent   = NL::min_exponent;
    static constexpr int  min_exponent10 = NL::min_exponent10;
    static constexpr int  max_exponent   = NL::max_exponent;
    static constexpr int  max_exponent10 = NL::max_exponent10;

    static constexpr bool has_infinity      = NL::has_infinity;
    static constexpr bool has_quiet_NaN     = NL::has_quiet_NaN;
    static constexpr bool has_signaling_NaN = NL::has_signaling_NaN;

    static constexpr bool is_iec559  = NL::is_iec559;
    static constexpr bool is_bounded = NL::is_bounded;
    static constexpr bool is_modulo  = NL::is_modulo;

    static constexpr bool              traps           = NL::traps;
    static constexpr bool              tinyness_before = NL::tinyness_before;
    static constexpr float_round_style round_style     = NL::round_style;
};

} // namespace std
