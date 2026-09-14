//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

// It isn't ideal putting all the sub tests into one test case, but approximately 150 different combinations of
// simd type and size might be tested, and the overhead of having one test case for each leads to compilation times
// in tens of minutes. Putting everything in one test case cuts this down dramatically.
BOOST_AUTO_TEST_CASE_TEMPLATE(BinaryOperators, TypeParam, NumericSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  using tvt = typename TypeParam::value_type;

  // Only integral division, including the user defined type, is tested here.
  constexpr bool doDivTests = std::integral<tvt> || std::same_as<tvt, UserDefinedInteger>;

  // For div/mod tests want a divisor which doesn't contain any zeros.
  const auto v0NotZero = applyUnary(f.v0, [](auto v) -> decltype(v) { return v == tvt() ? tvt(1) : v; });
  const auto v1NotZero = applyUnary(f.v1, [](auto v) -> decltype(v) { return v == tvt() ? tvt(1) : v; });

  test_binary_operator(f.v0, f.v1, std::plus<>());
  test_binary_operator(f.v0, f.v1, std::minus<>());
  test_binary_operator(f.v0, f.v1, std::multiplies<>());
  if constexpr (doDivTests)
  {
    test_binary_operator(v0NotZero, v1NotZero, std::modulus<>());
    test_binary_operator(v0NotZero, v1NotZero, std::divides<>()); // Float division is tricky to test. At least check integer.
  }

  auto assignedPlus = f.v0;
  assignedPlus += f.v1;
  BOOST_SIMD_EQUAL(assignedPlus, applyBinary(f.v0, f.v1, [](tvt x, tvt y) -> tvt { return tvt(x + y); }));

  auto assignedMinus = f.v0;
  assignedMinus -= f.v1;
  BOOST_SIMD_EQUAL(assignedMinus, applyBinary(f.v0, f.v1, [](tvt x, tvt y) -> tvt { return tvt(x - y); }));

  auto assignedMultiplies = f.v0;
  assignedMultiplies *= f.v1;
  BOOST_SIMD_EQUAL(assignedMultiplies, applyBinary(f.v0, f.v1, [](tvt x, tvt y) -> tvt { return tvt(x * y); }));

  if constexpr (doDivTests)
  {
    auto assignedModulus = f.v0;
    assignedModulus %= v1NotZero;
    BOOST_SIMD_EQUAL(assignedModulus, applyBinary(f.v0, v1NotZero, [](tvt x, tvt y) -> tvt { return tvt(x % y); }));

    // Float division is tricky to test. At least check integer division.
    auto assignedDivides = f.v0;
    assignedDivides /= v1NotZero;
    BOOST_SIMD_EQUAL(assignedDivides, applyBinary(f.v0, v1NotZero, [](tvt x, tvt y) -> tvt { return tvt(x / y); }));
  }
}

// :TODO: Division.

// :TODO: Binary division
// TYPED_TEST_P(SimdArithmeticTest, DISABLED_BinaryDivision) // :TODO: Figure out how to do this in integer too.
// {
// //   using tvt = typename TypeParam::value_type;

// //   // Can't have any zeros in the division. Replace zeros by 1.
// //   const auto nonzero = applyUnary (f.v1, [=](tvt v) { return (v ==  tvt()) ? tvt(1.0f) : v; });

// //   // Division is awkward to test as there may be different methods used with varying precision. So instead
// //   // of dividing one number by another, start my multiplying two known numbers together to get a new result
// //   // (which will be integer, because the test values are integer), and then divide the result to try to get
// //   // back to the original.
// //   const auto multiplied = applyBinary (f.v0, nonzero, [](tvt l, tvt r) { return l * r; });
// //   const auto computed = GetArray(round(multiplied / nonzero));

// //   EXPECT_EQ(computed, GetArray(f.v0));
// }
