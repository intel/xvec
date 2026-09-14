//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimdTestUtilities.hpp"

BOOST_AUTO_TEST_CASE(NumericLimits)
{
  using xvec::simd::vec;

  // ---- vec<float, 4> ----
  using VF = vec<float, 4>;
  using LF = std::numeric_limits<VF>;
  using SF = std::numeric_limits<float>;

  // Specialization is provided.
  static_assert(LF::is_specialized);

  // Trait members forwarded element-wise (same value as scalar).
  static_assert(LF::is_signed     == SF::is_signed);
  static_assert(LF::is_integer    == SF::is_integer);
  static_assert(LF::is_iec559     == SF::is_iec559);
  static_assert(LF::digits        == SF::digits);
  static_assert(LF::radix         == SF::radix);
  static_assert(LF::max_exponent  == SF::max_exponent);

  // Value-returning members return a vec, broadcast from the scalar limit.
  static_assert(std::is_same_v<decltype(LF::max()), VF>);
  static_assert(std::is_same_v<decltype(LF::min()), VF>);
  static_assert(std::is_same_v<decltype(LF::epsilon()), VF>);
  static_assert(std::is_same_v<decltype(LF::infinity()), VF>);

  // Element-wise equality with the broadcast scalar limit.
  BOOST_CHECK(all_of(LF::max()      == VF(SF::max())));
  BOOST_CHECK(all_of(LF::lowest()   == VF(SF::lowest())));
  BOOST_CHECK(all_of(LF::epsilon()  == VF(SF::epsilon())));
  BOOST_CHECK(all_of(LF::infinity() == VF(SF::infinity())));

  // ---- vec<int, 8> ----
  using VI = vec<int, 8>;
  using LI = std::numeric_limits<VI>;
  using SI = std::numeric_limits<int>;

  static_assert(LI::is_specialized);
  static_assert(LI::is_signed    == SI::is_signed);
  static_assert(LI::is_integer   == SI::is_integer);
  static_assert(LI::is_exact     == SI::is_exact);
  static_assert(LI::digits       == SI::digits);
  static_assert(LI::radix        == SI::radix);

  static_assert(std::is_same_v<decltype(LI::max()), VI>);
  static_assert(std::is_same_v<decltype(LI::min()), VI>);

  BOOST_CHECK(all_of(LI::max()    == VI(SI::max())));
  BOOST_CHECK(all_of(LI::min()    == VI(SI::min())));
  BOOST_CHECK(all_of(LI::lowest() == VI(SI::lowest())));

#if defined(__FLT16_MIN__)
  // ---- vec<_Float16, 8> ----
  using VH = vec<_Float16, 8>;
  using LH = std::numeric_limits<VH>;
  using SH = std::numeric_limits<_Float16>;

  static_assert(LH::is_specialized);
  static_assert(LH::is_signed     == SH::is_signed);
  static_assert(LH::is_integer    == SH::is_integer);
  static_assert(LH::is_iec559     == SH::is_iec559);
  static_assert(LH::digits        == SH::digits);
  static_assert(LH::radix         == SH::radix);
  static_assert(LH::max_exponent  == SH::max_exponent);

  static_assert(std::is_same_v<decltype(LH::max()), VH>);
  static_assert(std::is_same_v<decltype(LH::epsilon()), VH>);
  static_assert(std::is_same_v<decltype(LH::infinity()), VH>);

  BOOST_CHECK(all_of(LH::max()      == VH(SH::max())));
  BOOST_CHECK(all_of(LH::lowest()   == VH(SH::lowest())));
  BOOST_CHECK(all_of(LH::epsilon()  == VH(SH::epsilon())));
  BOOST_CHECK(all_of(LH::infinity() == VH(SH::infinity())));
#endif

  // ---- cv-qualified forms inherit from the unqualified specialization ----
  static_assert(std::numeric_limits<const VF>::is_specialized);
  static_assert(std::numeric_limits<volatile VF>::is_specialized);
  static_assert(std::numeric_limits<const volatile VF>::is_specialized);
  static_assert(std::numeric_limits<const VF>::digits == SF::digits);
}