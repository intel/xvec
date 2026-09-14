//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

template<typename SIMD, typename OP>
static void
checkBinaryOperator(SIMD lhs, SIMD rhs, OP op)
{
  BOOST_SIMD_EQUAL(op(lhs, rhs), applyBinaryWithSameTypes(lhs, rhs, op));
  BOOST_SIMD_EQUAL(op(rhs, lhs), applyBinaryWithSameTypes(rhs, lhs, op));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BinaryBitwiseOperatorSimd, TypeParam, BitmaskSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  test_binary_operator(f.v0, f.v1, std::bit_and<>());
  test_binary_operator(f.v0, f.v1, std::bit_or<>());
  test_binary_operator(f.v0, f.v1, std::bit_xor<>());

  auto assignedAnd = f.v0;
  assignedAnd &= f.v1;
  BOOST_SIMD_EQUAL(assignedAnd, applyBinaryWithSameTypes(f.v0, f.v1, std::bit_and<>()));

  auto assignedOr = f.v0;
  assignedOr |= f.v1;
  BOOST_SIMD_EQUAL(assignedOr, applyBinaryWithSameTypes(f.v0, f.v1, std::bit_or<>()));

  auto assignedXor = f.v0;
  assignedXor ^= f.v1;
  BOOST_SIMD_EQUAL(assignedXor, applyBinaryWithSameTypes(f.v0, f.v1, std::bit_xor<>()));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(Complement, TypeParam, BitmaskSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnaryWithSameTypes(f.v0, [](tvt x) -> tvt { return tvt(~x); });
  BOOST_SIMD_EQUAL(~(f.v0), expected);
}

