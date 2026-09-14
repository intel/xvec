//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

#include <functional>

/// FP class operations (finite, denormal, inf).
template<typename T> class SimdFpclassFixture {
public:
  const typename T::value_type k_nan = (typename T::value_type)std::numeric_limits<float>::quiet_NaN();
  const typename T::value_type k_infinity = (typename T::value_type)std::numeric_limits<float>::infinity();
  const typename T::value_type k_sqrtMinusOne = (typename T::value_type)sqrt(-1.0);
  const typename T::value_type k_zero = {};
  const typename T::value_type k_denormal = [](){
    union {
      typename T::value_type lowest_denormal;
      int64_t i;
    };
    i = 1; // This will generate the lowest possible floating point value in any width.
    return lowest_denormal;
  }();
};

/// Given a classification function (e.g., isnan) apply it to each of the valid values and invalid values
/// in turn to check that the appropriate true/false response is returned.
template <typename TypeParam, typename VALID_CONTAINER, typename INVALID_CONTAINER>
static void TestFpclass(std::function<typename TypeParam::mask_type(TypeParam)> classify_fn,
                        const VALID_CONTAINER& valid_values,
                        const INVALID_CONTAINER& invalid_values)
{
  for (auto x : valid_values)
    BOOST_TEST(classify_fn(TypeParam(x)).to_bitset().all() == true);

  for (auto x : invalid_values)
    BOOST_TEST(classify_fn(TypeParam(x)).to_bitset().none() == true);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IsInfinite, TypeParam, FloatSimdTypes)
{
  SimdFpclassFixture<TypeParam> f;

  typename TypeParam::value_type inf_values[] = {
    f.k_infinity, -f.k_infinity
  };

  typename TypeParam::value_type noninf_values[] = {
    -99, -1, 0, 1, 99, -f.k_nan, f.k_nan,
  };

  TestFpclass<TypeParam>([](TypeParam x) { return isinf(x); }, inf_values, noninf_values);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IsNan, TypeParam, FloatSimdTypes)
{
  SimdFpclassFixture<TypeParam> f;

  typename TypeParam::value_type nan_values[] = {
    f.k_sqrtMinusOne, -f.k_sqrtMinusOne
  };

  typename TypeParam::value_type nonnan_values[] = {
    -99, -1, 0, 1, 99, f.k_infinity, -f.k_infinity,
  };

  TestFpclass<TypeParam>([](TypeParam x) { return isnan(x); }, nan_values, nonnan_values);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IsFinite, TypeParam, FloatSimdTypes)
{
  SimdFpclassFixture<TypeParam> f;

  typename TypeParam::value_type finite_values[] = {
    -99, -1, -0, 0, 1, 99
  };

  typename TypeParam::value_type nonfinite_values[] = {
    f.k_infinity, -f.k_infinity, -f.k_nan, f.k_nan,
  };

  TestFpclass<TypeParam>([](TypeParam x) { return isfinite(x); }, finite_values, nonfinite_values);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IsNormal, TypeParam, FloatSimdTypes)
{
  SimdFpclassFixture<TypeParam> f;

  typename TypeParam::value_type normal_values[] = {
    -999, -3, -2, -1, 1, 2, 3, 999,
#if defined(__FLT16_MIN__)
    -__FLT16_MIN__,  __FLT16_MIN__, -__FLT16_MAX__,  __FLT16_MAX__,
#endif
  };

  typename TypeParam::value_type nonnormal_values[] = {
    -f.k_zero, f.k_zero, -f.k_denormal, f.k_denormal,
    -f.k_nan, f.k_nan, -f.k_infinity, f.k_infinity
  };

  TestFpclass<TypeParam>([](TypeParam x) { return isnormal(x); }, normal_values, nonnormal_values);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SignbitFp, TypeParam, FloatSimdTypes)
{
  using namespace xvec::simd::detail;

  using _Tp = typename TypeParam::value_type;
  SimdFpclassFixture<TypeParam> f;

  _Tp negative_values[] = {
    -f.k_nan, -f.k_infinity, -f.k_denormal, -f.k_zero, -10, -10000, -std::numeric_limits<_Tp>::min(), -std::numeric_limits<_Tp>::max()
  };

  _Tp positive_values[] = {
    f.k_nan, f.k_infinity, f.k_denormal, f.k_zero, 10, 10000, std::numeric_limits<_Tp>::min(), std::numeric_limits<_Tp>::max()
  };

  TestFpclass<TypeParam>([](TypeParam x) { return signbit(x); }, negative_values, positive_values);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SignbitInteger, TypeParam, IntegerSimdTypes)
{
  using _Tp = typename TypeParam::value_type;
  SimdTestFixture<TypeParam> f;

  auto expected = typename TypeParam::mask_type([=](auto i) { return std::signbit(f.v0[i]); });
  BOOST_TEST(signbit(f.v0).to_bitset() == expected.to_bitset());

  // Explicit set the MSB to ensure that unsigned types definitely get an MSB
  // to check (otherwise it is random whether a sufficiently large unsigned
  // value is tested).
  BOOST_TEST(signbit(TypeParam(std::numeric_limits<_Tp>::max())).to_bitset() == (typename TypeParam::mask_type()).to_bitset());

}
