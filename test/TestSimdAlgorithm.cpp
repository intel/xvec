//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(Minimum, TypeParam, OrderableSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  const auto expected = applyBinary(f.v0, f.v1, [](auto v0, auto v1) { return std::min(v0, v1); });

  BOOST_SIMD_EQUAL(min(f.v0, f.v1), expected);
  BOOST_SIMD_EQUAL(min(f.v1, f.v0), expected);
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprMinimum, TypeParam, ArithmeticSimdTypes)
{
  constexpr auto v0 = GetConstexprRandomVector<TypeParam, 0>();
  constexpr auto v1 = GetConstexprRandomVector<TypeParam, 1>();

  const auto expected = applyBinary(v0, v1, [](auto v0, auto v1) { return std::min(v0, v1); });

  constexpr auto m0 = min(v0, v1);
  constexpr auto m1 = min(v1, v0);

  BOOST_SIMD_EQUAL(m0, expected);
  BOOST_SIMD_EQUAL(m1, expected);
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(Maximum,  TypeParam, OrderableSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  const auto expected = applyBinary(f.v0, f.v1, [](auto v0, auto v1) { return std::max(v0, v1); });

  BOOST_SIMD_EQUAL(max(f.v0, f.v1), expected);
  BOOST_SIMD_EQUAL(max(f.v1, f.v0), expected);
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprMaximum, TypeParam, ArithmeticSimdTypes)
{
  constexpr auto v0 = GetConstexprRandomVector<TypeParam, 0>();
  constexpr auto v1 = GetConstexprRandomVector<TypeParam, 1>();

  const auto expected = applyBinary(v0, v1, [](auto v0, auto v1) { return std::max(v0, v1); });

  constexpr auto m0 = max(v0, v1);
  constexpr auto m1 = max(v1, v0);

  BOOST_SIMD_EQUAL(m0, expected);
  BOOST_SIMD_EQUAL(m1, expected);
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(MinMax, TypeParam, OrderableSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  const auto expectedMin = applyBinary(f.v0, f.v1,[](auto v0, auto v1) { return std::min(v0, v1); });
  const auto expectedMax = applyBinary(f.v0, f.v1, [](auto v0, auto v1) { return std::max(v0, v1); });

  const auto [minComputed, maxComputed] = minmax(f.v0, f.v1);

  BOOST_SIMD_EQUAL(minComputed, expectedMin);
  BOOST_SIMD_EQUAL(maxComputed, expectedMax);
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprMinMax, TypeParam, ArithmeticSimdTypes)
{
  constexpr auto v0 = GetConstexprRandomVector<TypeParam, 0>();
  constexpr auto v1 = GetConstexprRandomVector<TypeParam, 1>();

  const auto expectedMin = applyBinary(v0, v1, [](auto v0, auto v1) { return std::min(v0, v1); });
  const auto expectedMax = applyBinary(v0, v1, [](auto v0, auto v1) { return std::max(v0, v1); });

  constexpr auto mm = minmax(v0, v1);

  BOOST_SIMD_EQUAL(mm.first, expectedMin);
  BOOST_SIMD_EQUAL(mm.second, expectedMax);
}
#endif

// :TODO: Clamp
