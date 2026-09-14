//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

template<typename T> struct IsLogicalNotable : std::bool_constant<requires(typename T::value_type a) { !a; }> {};
using LogicalNotableTypes = boost::mp11::mp_copy_if<AllSimdTypes, IsLogicalNotable>;

/// Utility function to return a value which is different to what is supplied,
/// which is used to generate unique simd values from arbitrary types.
template<typename _Tp>
static _Tp getDifferentValue(_Tp in)
{
  if constexpr (std::is_enum_v<_Tp>) return _Tp(int(in) ^ 1); // Assume a test enum which has at least 0 and 1 values, so flipping the lowest bit makes it unique.
  else return _Tp(in + _Tp(1));
}

// More rigorous test for main equality
BOOST_AUTO_TEST_CASE_TEMPLATE(Equality, TypeParam, AllSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // Create a new vector which is known to be different to v0.
  alignas(64) typename SimdTestFixture<TypeParam>::test_array_type differentToV0Data;
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    differentToV0Data[i] = getDifferentValue(f.v0[i]);
  const auto differentToV0 = xvec::simd::partial_load<TypeParam>(differentToV0Data);

  BOOST_TEST((f.v0 == f.v0).to_bitset().all() == true);
  BOOST_TEST((f.v0 != differentToV0).to_bitset().all() == true);
  BOOST_TEST((differentToV0 != f.v0).to_bitset().all() == true);

  // Create a more random bitmask of equal and not-equal elements to check for other patterns.
  const auto mask = typename TypeParam::mask_type(getRandomBitset<TypeParam::size()>());
  TypeParam randomBlend([=](auto i) {
    return mask.to_bitset().test(i) ? f.v0[i] : differentToV0[i];
  });

  BOOST_TEST((f.v0 == randomBlend).to_bitset() == mask.to_bitset());
  BOOST_TEST((f.v0 != randomBlend).to_bitset() == ~mask.to_bitset());
}

template<typename SIMD, typename CMP_OP>
constexpr static void
checkCompare(SIMD lhs, SIMD rhs, typename SIMD::mask_type l_to_r, typename SIMD::mask_type r_to_l, CMP_OP comparison)
{
  constexpr auto numBits = SIMD::size();
  using bitmask = std::bitset<numBits>;

  const auto lhsArray = to_array(lhs);
  const auto rhsArray = to_array(rhs);

  // Test ordering the operands LHS OP RHS and RHS OP LHS
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

BOOST_AUTO_TEST_CASE_TEMPLATE(SimdComparisonOperators, TypeParam, OrderableSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  auto eqL = (f.v0 == f.v1);
  auto eqR = (f.v1 == f.v0);
  checkCompare(f.v0, f.v1, eqL, eqR, std::equal_to<>());

  auto neqL = (f.v0 != f.v1);
  auto neqR = (f.v1 != f.v0);
  checkCompare(f.v0, f.v1, neqL, neqR, std::not_equal_to<>());

  auto lessL = (f.v0 < f.v1);
  auto lessR = (f.v1 < f.v0);
  checkCompare(f.v0, f.v1, lessL, lessR, std::less<>());

  auto leqL = (f.v0 <= f.v1);
  auto leqR = (f.v1 <= f.v0);
  checkCompare(f.v0, f.v1, leqL, leqR, std::less_equal<>());

  auto greaterL = (f.v0 > f.v1);
  auto greaterR = (f.v1 > f.v0);
  checkCompare(f.v0, f.v1, greaterL, greaterR, std::greater<>());

  auto geqL = (f.v0 >= f.v1);
  auto geqR = (f.v1 >= f.v0);
  checkCompare(f.v0, f.v1, geqL, geqR, std::greater_equal<>());
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprSimdComparisonOperators, TypeParam, ArithmeticSimdTypes)
{
  constexpr auto v0 = GetConstexprRandomVector<TypeParam, 0>();
  constexpr auto v1 = GetConstexprRandomVector<TypeParam, 1>();

  constexpr auto eqL = (v0 == v1);
  constexpr auto eqR = (v1 == v0);
  checkCompare(v0, v1, eqL, eqR, std::equal_to<>());

  constexpr auto neqL = (v0 != v1);
  constexpr auto neqR = (v1 != v0);
  checkCompare(v0, v1, neqL, neqR, std::not_equal_to<>());

  constexpr auto lessL = (v0 < v1);
  constexpr auto lessR = (v1 < v0);
  checkCompare(v0, v1, lessL, lessR, std::less<>());

  constexpr auto leqL = (v0 <= v1);
  constexpr auto leqR = (v1 <= v0);
  checkCompare(v0, v1, leqL, leqR, std::less_equal<>());

  constexpr auto greaterL = (v0 > v1);
  constexpr auto greaterR = (v1 > v0);
  checkCompare(v0, v1, greaterL, greaterR, std::greater<>());

  constexpr auto geqL = (v0 >= v1);
  constexpr auto geqR = (v1 >= v0);
  checkCompare(v0, v1, geqL, geqR, std::greater_equal<>());
}
#endif

// Similar to previous, but do comparison with a scalar picked out of the vector instead. This
// checks that the scalar operator comparison works, and gives an output which is fairly balanced
// between set and cleared bits.
template<typename SIMD, typename CMP_OP>
static void
checkScalarCompare(typename SIMD::value_type scalar, SIMD value, CMP_OP comparison,
                   typename SIMD::mask_type l2rmask, typename SIMD::mask_type r2lmask)
{
  using bitmask = std::bitset<SIMD::size()>;

  const auto array = to_array(value);

  // Test both orders of the scalar and vector
  bitmask expectedScalarLeft;
  bitmask expectedScalarRight;
  for (std::size_t i=0; i<SIMD::size(); ++i)
  {
    expectedScalarLeft[i] = comparison(scalar, array[i]);
    expectedScalarRight[i] = comparison(array[i], scalar);
  }

  BOOST_TEST(expectedScalarLeft == l2rmask.to_bitset());
  BOOST_TEST(expectedScalarRight == r2lmask.to_bitset());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ScalarComparisonOperators, TypeParam, OrderableSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // For equality tests, pick the first value as that ensures that at least one
  // element will compare equal or not equal since it definitely in the simd.
  const auto scalar = f.v0[0];
  checkScalarCompare(scalar, f.v0, std::equal_to<>(),      scalar == f.v0, f.v0 == scalar);
  checkScalarCompare(scalar, f.v0, std::not_equal_to<>(),  scalar != f.v0, f.v0 != scalar);

  const auto median = getMedian(f.v0); // Choose something in the middle of the number range so that other values are less or greater than this value.
  checkScalarCompare(median, f.v0, std::less<>(),          median <  f.v0, f.v0 <  median);
  checkScalarCompare(median, f.v0, std::less_equal<>(),    median <= f.v0, f.v0 <= median);
  checkScalarCompare(median, f.v0, std::greater<>(),       median >  f.v0, f.v0 >  median);
  checkScalarCompare(median, f.v0, std::greater_equal<>(), median >= f.v0, f.v0 >= median);
}

// Sort of a compare, since it compares against true.
BOOST_AUTO_TEST_CASE_TEMPLATE(Not, TypeParam, LogicalNotableTypes)
{
  SimdTestFixture<TypeParam> f;
  SimdMaskFixture<typename TypeParam::mask_type> b;

  // Use the mask to put some zeroes into the value to test. Without this, a
  // test for some wide ranging type (e.g., double) will almost never have false
  // values if they are picked randomly. The bit mask will have 50% probably of
  // zeroes though, making this test more reliable.
  TypeParam input([=](auto i) {
    return typename TypeParam::value_type(b.bitset0[i] ? f.v0[i] : 0);
  });

  std::bitset<TypeParam::size> expected = {};
  for (int i=0; i<TypeParam::size(); ++i)
    expected[i] = !input[i];

  BOOST_TEST((!input).to_bitset() == expected);
}
