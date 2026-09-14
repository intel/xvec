//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

// Overloads needed for FP16 since the default library doesn't provide the
// comparison functions for _Float16 yet. Each comparison is computed in float.
#if defined(__FLT16_MIN__)
namespace xvec {
bool isgreater(_Float16 x, _Float16 y)      { return std::isgreater(float(x), float(y)); }
bool isgreaterequal(_Float16 x, _Float16 y) { return std::isgreaterequal(float(x), float(y)); }
bool isless(_Float16 x, _Float16 y)         { return std::isless(float(x), float(y)); }
bool islessequal(_Float16 x, _Float16 y)    { return std::islessequal(float(x), float(y)); }
bool islessgreater(_Float16 x, _Float16 y)  { return std::islessgreater(float(x), float(y)); }
bool isunordered(_Float16 x, _Float16 y)    { return std::isunordered(float(x), float(y)); }
}
#endif

/// Check a simd comparison-to-mask function against a scalar reference applied
/// element-wise. Both orderings (x OP y and y OP x) are tested, mirroring the
/// relational operator tests in TestSimdCompare.cpp.
template<typename SIMD, typename CMP_OP>
static void
checkMathCompare(SIMD lhs, SIMD rhs, typename SIMD::mask_type l_to_r,
                 typename SIMD::mask_type r_to_l, CMP_OP comparison)
{
  constexpr auto numBits = SIMD::size();
  using bitmask = std::bitset<numBits>;

  const auto lhsArray = to_array(lhs);
  const auto rhsArray = to_array(rhs);

  bitmask expectedLeftToRight;
  bitmask expectedRightToLeft;
  for (std::size_t i=0; i<numBits; ++i)
  {
    expectedLeftToRight[i] = comparison(lhsArray[i], rhsArray[i]);
    expectedRightToLeft[i] = comparison(rhsArray[i], lhsArray[i]);
  }

  BOOST_TEST(expectedLeftToRight == l_to_r.to_bitset());
  BOOST_TEST(expectedRightToLeft == r_to_l.to_bitset());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MathCompareToMask, TypeParam, FloatSimdTypes)
{
  using std::isgreater;
  using std::isgreaterequal;
  using std::isless;
  using std::islessequal;
  using std::islessgreater;
  using std::isunordered;

  using xvec::isgreater;
  using xvec::isgreaterequal;
  using xvec::isless;
  using xvec::islessequal;
  using xvec::islessgreater;
  using xvec::isunordered;

  SimdTestFixture<TypeParam> f;

  // Reference scalar lambdas use the std:: equivalents (or the FP16 overloads
  // above). Note that for ordered, non-NaN data these match the relational
  // operators exactly, which is what the random test data produces.
  checkMathCompare(f.v0, f.v1, isgreater(f.v0, f.v1), isgreater(f.v1, f.v0),
                   [](auto x, auto y) { return isgreater(x, y); });

  checkMathCompare(f.v0, f.v1, isgreaterequal(f.v0, f.v1), isgreaterequal(f.v1, f.v0),
                   [](auto x, auto y) { return isgreaterequal(x, y); });

  checkMathCompare(f.v0, f.v1, isless(f.v0, f.v1), isless(f.v1, f.v0),
                   [](auto x, auto y) { return isless(x, y); });

  checkMathCompare(f.v0, f.v1, islessequal(f.v0, f.v1), islessequal(f.v1, f.v0),
                   [](auto x, auto y) { return islessequal(x, y); });

  checkMathCompare(f.v0, f.v1, islessgreater(f.v0, f.v1), islessgreater(f.v1, f.v0),
                   [](auto x, auto y) { return islessgreater(x, y); });

  checkMathCompare(f.v0, f.v1, isunordered(f.v0, f.v1), isunordered(f.v1, f.v0),
                   [](auto x, auto y) { return isunordered(x, y); });
}