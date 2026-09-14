//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cfenv>

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

/// Rounding operations.
template<typename T> class SimdRoundingFixture {
public:
  using test_value_type = typename T::value_type;

  //  For the rounding tests create a vector of values which are evenly spaced over a
  //  range. Note that the values are perturbed slightly off the boundary (i.e., deliberately
  //  avoid 0.5, 1.5, 2.5, etc) since those values are handled very slightly differently under some
  // conditions. For example, std::round(0.5) gives 1.0, while roundscale(0.5) gives 0.0.
  T roundV = T([=](auto i) {
    return test_value_type((8.0 * float(i)) / 32.0f - 3.9f);
  });
  T tieV = T([=](auto i) {
    constexpr std::array<float, 8> values{-3.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 3.5f};
    return test_value_type(values[int(i) & 7]);
  });

};

class ScopedRoundingMode {
public:
  explicit ScopedRoundingMode(int newMode) : previousMode(std::fegetround())
  {
    BOOST_REQUIRE(previousMode != -1);
    BOOST_REQUIRE_EQUAL(std::fesetround(newMode), 0);
  }

  ~ScopedRoundingMode() { BOOST_CHECK_EQUAL(std::fesetround(previousMode), 0); }

private:
  int previousMode;
};

// Apply the FP32 float rounding function to whatever data is supplied. In the case of complex values
// apply the rounding to each individual value. Note that the incoming rnd function is FP32 (ceilf,
// truncf, etc) but the test data is representable in every float type so a conversion to and from
// FP32 won't harm the data.
template<typename T, typename FN> static T expectedRounding(T v, FN rnd_fn) {
  return T(rnd_fn((float(v))));
}
template<typename T, typename FN> static std::complex<T> expectedRounding(std::complex<T> v, FN rnd_fn) {
  return std::complex<T>(expectedRounding(v.real(), rnd_fn), expectedRounding(v.imag(), rnd_fn));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Ceil, TypeParam, FloatSimdTypes)
{
  SimdRoundingFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary (f.roundV, [](tvt value) { return expectedRounding(value, ceilf); } );
  BOOST_SIMD_EQUAL(ceil(f.roundV), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Floor, TypeParam, FloatSimdTypes)
{
  SimdRoundingFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary (f.roundV, [](tvt value) { return expectedRounding(value, floorf); });
  BOOST_SIMD_EQUAL(floor(f.roundV), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Truncate, TypeParam, FloatSimdTypes)
{
  SimdRoundingFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary (f.roundV, [](tvt value) { return expectedRounding(value, truncf); });
  BOOST_SIMD_EQUAL(trunc(f.roundV), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Round, TypeParam, FloatSimdTypes)
{
  SimdRoundingFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary (f.roundV, [](tvt value) { return expectedRounding(value, roundf); });
  BOOST_SIMD_EQUAL(round(f.roundV), expected);

  const auto expectedTies = applyUnary(f.tieV, [](tvt value) { return expectedRounding(value, roundf); });
  BOOST_SIMD_EQUAL(round(f.tieV), expectedTies);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(RintNearest, TypeParam, FloatSimdTypes)
{
  ScopedRoundingMode roundingMode(FE_TONEAREST);
  SimdRoundingFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.tieV, [](tvt value) {
    return expectedRounding(value, [](float x) { return std::rint(x); });
  });
  BOOST_SIMD_EQUAL(rint(f.tieV), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(RintDownward, TypeParam, FloatSimdTypes)
{
  ScopedRoundingMode roundingMode(FE_DOWNWARD);
  SimdRoundingFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.roundV, [](tvt value) {
    return expectedRounding(value, [](float x) { return std::rint(x); });
  });
  BOOST_SIMD_EQUAL(rint(f.roundV), expected);
}
