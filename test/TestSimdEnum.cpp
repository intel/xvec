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

using namespace xvec;

BOOST_AUTO_TEST_CASE_TEMPLATE(ToUnderlying, TypeParam, EnumSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  using _Tp = typename TypeParam::value_type;
  using _Under = std::underlying_type_t<_Tp>;

  const auto expected = applyUnary (f.v0, [](auto v) -> _Under { return static_cast<_Under>(v); });
  const auto computed = to_underlying(f.v0);
  BOOST_SIMD_EQUAL(computed, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckDisallowedScopedOperators, TypeParam, ScopedEnumSimdTypes)
{
  // Unary arithmetic
  static_assert(!std::invocable<std::negate<>, TypeParam>);
  static_assert(!std::invocable<std::bit_not<>, TypeParam>);

  // Binary arithmetic
  static_assert(!std::invocable<std::plus<>, TypeParam, TypeParam>);
  static_assert(!std::invocable<std::minus<>, TypeParam, TypeParam>);
  static_assert(!std::invocable<std::multiplies<>, TypeParam, TypeParam>);
  static_assert(!std::invocable<std::divides<>, TypeParam, TypeParam>);
  static_assert(!std::invocable<std::modulus<>, TypeParam, TypeParam>);

  // Increment/decrement
  static_assert(!requires (TypeParam v) { ++v; });
  static_assert(!requires (TypeParam v) { v++; });
  static_assert(!requires (TypeParam v) { --v; });
  static_assert(!requires (TypeParam v) { v--; });

  // Shifts
  static_assert(!requires (TypeParam v) { v << 1; });
  static_assert(!requires (TypeParam v) { v >> 1; });
  static_assert(!requires (TypeParam v) { v << v; });
  static_assert(!requires (TypeParam v) { v >> v; });
  static_assert(!requires (TypeParam v) { v <<= 1; });
  static_assert(!requires (TypeParam v) { v >>= 1; });
  static_assert(!requires (TypeParam v) { v <<= v; });
  static_assert(!requires (TypeParam v) { v >>= v; });
}
