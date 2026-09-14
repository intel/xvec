//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(Fma, TypeParam, FloatSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  const auto expected = applyTernary(f.v0, f.v1, f.v2, [](auto v0, auto v1, auto v2) {
     return v0 * v1 + v2; });

  BOOST_SIMD_EQUAL(fma(f.v0, f.v1, f.v2), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FmAddSub, TypeParam, FloatSimdTypes)
{
  typename SimdTestFixture<TypeParam>::test_array_type expected;
  SimdTestFixture<TypeParam> f;

  for (size_t i=0; i<expected.size(); ++i)
  {
    // Evens subtract, odds add.
    expected[i] = (i % 2 == 0)
      ? f.v0[i] * f.v1[i] - f.v2[i]
      : f.v0[i] * f.v1[i] + f.v2[i];
  }

  BOOST_TEST(to_array(fmaddsub(f.v0, f.v1, f.v2)) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FmSubAdd, TypeParam, FloatSimdTypes)
{
  typename SimdTestFixture<TypeParam>::test_array_type expected;
  SimdTestFixture<TypeParam> f;

  for (size_t i=0; i<expected.size(); ++i)
  {
    // Evens add, odds subtract
    expected[i] = (i % 2 == 0)
      ? f.v0[i] * f.v1[i] + f.v2[i]
      : f.v0[i] * f.v1[i] - f.v2[i];
  }

  BOOST_TEST(to_array(fmsubadd(f.v0, f.v1, f.v2)) == expected, boost::test_tools::per_element());
}
