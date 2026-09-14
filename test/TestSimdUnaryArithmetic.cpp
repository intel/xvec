//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

// Remove any simd type which can't have ++/-- applied to it.
template<typename _T> struct IsIncrementable { static constexpr bool value = std::incrementable<typename _T::value_type>; };
using IncrementableTypes = boost::mp11::mp_copy_if<AllSimdTypes, IsIncrementable>;

BOOST_AUTO_TEST_CASE_TEMPLATE(UnaryNegate, TypeParam, NumericSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [](tvt x) -> tvt { return tvt(-x); });
  BOOST_SIMD_EQUAL(-f.v0, expected);

  #if defined(_XVEC_HAS_CONSTEXPR)
    // constexpr variant.
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 0>(5);
    constexpr auto negCv = -cv;
    BOOST_SIMD_EQUAL(negCv, -cv);
  #endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(UnaryPlus, TypeParam, NumericSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [](tvt x) -> tvt { return x; });
  BOOST_SIMD_EQUAL(+f.v0, expected);

  // constexpr variant.
  constexpr auto cv = GetConstexprRandomVector<TypeParam, 0>(5);
  constexpr auto negCv = +cv;
  BOOST_SIMD_EQUAL(negCv, +cv);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(PreIncrement, TypeParam, IncrementableTypes)
{
  SimdTestFixture<TypeParam> f;

  auto computed_pre = f.v0;
  const auto new_value = ++computed_pre;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [](tvt x) -> tvt { return x + 1; });

  BOOST_SIMD_EQUAL(new_value, expected);
  BOOST_SIMD_EQUAL(computed_pre, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(PostIncrement, TypeParam, IncrementableTypes)
{
  SimdTestFixture<TypeParam> f;

  auto computed_post = f.v0;
  const auto new_value = computed_post++;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [](tvt x) -> tvt { return x + 1; });

  BOOST_SIMD_EQUAL(new_value, f.v0);
  BOOST_SIMD_EQUAL(computed_post, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(PreDecrement, TypeParam, IncrementableTypes)
{
  SimdTestFixture<TypeParam> f;

  auto computed_pre = f.v0;
  const auto new_value = --computed_pre;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [](tvt x) -> tvt { return x - 1; });

  BOOST_SIMD_EQUAL(new_value, expected);
  BOOST_SIMD_EQUAL(computed_pre, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(PostDecrement, TypeParam, IncrementableTypes)
{
  SimdTestFixture<TypeParam> f;

  auto computed_post = f.v0;
  const auto new_value = computed_post--;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [](tvt x) -> tvt { return x - 1; });

  BOOST_SIMD_EQUAL(new_value, f.v0);
  BOOST_SIMD_EQUAL(computed_post, expected);
}
