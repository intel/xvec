//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"
#include <xvec/detail/compat.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

/// Loads repeating values from an array.
template<typename Simd, typename T, std::size_t N>
constexpr Simd makeSimdFromValues(const std::array<T, N>& values)
{
  return Simd([&](auto i) { return values[i % N]; });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SatAdd, TypeParam, IntegerSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  constexpr auto lowest = std::numeric_limits<_Tp>::min();
  constexpr auto highest = std::numeric_limits<_Tp>::max();

  constexpr auto expected_add = [](_Tp lhs, _Tp rhs) {
    return xvec::simd::detail::saturating_add(lhs, rhs);
  };

  // Different elements exercise ordinary addition, zero, both endpoints,
  // upper saturation, and both operand orders.
  constexpr std::array lhsValues{
    _Tp(20),
    _Tp(0),
    lowest,
    highest,
    highest,
    _Tp(1),
  };

  constexpr std::array rhsValues{
    _Tp(22),
    _Tp(0),
    _Tp(0),
    _Tp(0),
    _Tp(1),
    highest,
  };

  const auto lhs = makeSimdFromValues<TypeParam>(lhsValues);
  const auto rhs = makeSimdFromValues<TypeParam>(rhsValues);

  BOOST_TEST(
    to_array(saturating_add(lhs, rhs)) ==
      applyBinaryToArray(lhs, rhs, expected_add),
    boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto constexprLhs =
      makeSimdFromValues<TypeParam>(lhsValues);
    constexpr auto constexprRhs =
      makeSimdFromValues<TypeParam>(rhsValues);

    static_assert(
      to_array(saturating_add(constexprLhs, constexprRhs)) ==
      applyBinaryToArray(constexprLhs, constexprRhs, expected_add));
  }
#endif

  if constexpr (std::signed_integral<_Tp>) {
    // Exercise lower saturation, values adjacent to both endpoints, and
    // ordinary addition of operands with opposite signs.
    constexpr std::array signedLhsValues{
      lowest,
      _Tp(-1),
      highest,
      lowest,
      _Tp(42),
      _Tp(-20),
    };

    constexpr std::array signedRhsValues{
      _Tp(-1),
      lowest,
      _Tp(-1),
      _Tp(1),
      _Tp(-20),
      _Tp(42),
    };

    const auto signedLhs =
      makeSimdFromValues<TypeParam>(signedLhsValues);
    const auto signedRhs =
      makeSimdFromValues<TypeParam>(signedRhsValues);

    BOOST_TEST(
      to_array(saturating_add(signedLhs, signedRhs)) ==
        applyBinaryToArray(signedLhs, signedRhs, expected_add),
      boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto constexprLhs =
        makeSimdFromValues<TypeParam>(signedLhsValues);
      constexpr auto constexprRhs =
        makeSimdFromValues<TypeParam>(signedRhsValues);

      static_assert(
        to_array(saturating_add(constexprLhs, constexprRhs)) ==
        applyBinaryToArray(
          constexprLhs, constexprRhs, expected_add));
    }
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SatSub, TypeParam, IntegerSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  constexpr auto lowest = std::numeric_limits<_Tp>::min();
  constexpr auto highest = std::numeric_limits<_Tp>::max();

  constexpr auto expected_sub = [](_Tp lhs, _Tp rhs) {
    return xvec::simd::detail::saturating_sub(lhs, rhs);
  };

  // Different elements exercise ordinary subtraction, zero, exact
  // endpoints, and lower saturation.
  constexpr std::array lhsValues{
    _Tp(42),
    _Tp(42),
    lowest,
    highest,
    highest,
    lowest,
  };

  constexpr std::array rhsValues{
    _Tp(20),
    _Tp(42),
    _Tp(0),
    _Tp(0),
    highest,
    _Tp(1),
  };

  const auto lhs = makeSimdFromValues<TypeParam>(lhsValues);
  const auto rhs = makeSimdFromValues<TypeParam>(rhsValues);

  BOOST_TEST(
    to_array(saturating_sub(lhs, rhs)) ==
      applyBinaryToArray(lhs, rhs, expected_sub),
    boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto constexprLhs =
      makeSimdFromValues<TypeParam>(lhsValues);
    constexpr auto constexprRhs =
      makeSimdFromValues<TypeParam>(rhsValues);

    static_assert(
      to_array(saturating_sub(constexprLhs, constexprRhs)) ==
      applyBinaryToArray(constexprLhs, constexprRhs, expected_sub));
  }
#endif

  if constexpr (std::signed_integral<_Tp>) {
    // Exercise upper saturation, values adjacent to both endpoints, and
    // ordinary subtraction involving negative values.
    constexpr std::array signedLhsValues{
      highest,
      lowest,
      highest,
      _Tp(20),
      _Tp(-20),
    };

    constexpr std::array signedRhsValues{
      _Tp(-1),
      _Tp(-1),
      _Tp(1),
      _Tp(-22),
      _Tp(22),
    };

    const auto signedLhs =
      makeSimdFromValues<TypeParam>(signedLhsValues);
    const auto signedRhs =
      makeSimdFromValues<TypeParam>(signedRhsValues);

    BOOST_TEST(
      to_array(saturating_sub(signedLhs, signedRhs)) ==
        applyBinaryToArray(signedLhs, signedRhs, expected_sub),
      boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto constexprLhs =
        makeSimdFromValues<TypeParam>(signedLhsValues);
      constexpr auto constexprRhs =
        makeSimdFromValues<TypeParam>(signedRhsValues);

      static_assert(
        to_array(saturating_sub(constexprLhs, constexprRhs)) ==
        applyBinaryToArray(
          constexprLhs, constexprRhs, expected_sub));
    }
#endif
  } else {
    // Exercise unsigned underflow, exact zero, and a representable value
    // adjacent to the upper endpoint.
    constexpr std::array unsignedLhsValues{
      _Tp(0),
      _Tp(1),
      highest,
    };

    constexpr std::array unsignedRhsValues{
      _Tp(1),
      _Tp(1),
      _Tp(1),
    };

    const auto unsignedLhs =
      makeSimdFromValues<TypeParam>(unsignedLhsValues);
    const auto unsignedRhs =
      makeSimdFromValues<TypeParam>(unsignedRhsValues);

    BOOST_TEST(
      to_array(saturating_sub(unsignedLhs, unsignedRhs)) ==
        applyBinaryToArray(
          unsignedLhs, unsignedRhs, expected_sub),
      boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto constexprLhs =
        makeSimdFromValues<TypeParam>(unsignedLhsValues);
      constexpr auto constexprRhs =
        makeSimdFromValues<TypeParam>(unsignedRhsValues);

      static_assert(
        to_array(saturating_sub(constexprLhs, constexprRhs)) ==
        applyBinaryToArray(
          constexprLhs, constexprRhs, expected_sub));
    }
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SatMul, TypeParam, IntegerSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  constexpr auto lowest = std::numeric_limits<_Tp>::min();
  constexpr auto highest = std::numeric_limits<_Tp>::max();

  constexpr auto expected_mul = [](_Tp lhs, _Tp rhs) {
    return xvec::simd::detail::saturating_mul(lhs, rhs);
  };

  // Different elements exercise ordinary multiplication, zero, identity,
  // upper saturation, and both operand orders.
  constexpr std::array lhsValues{
    _Tp(6),
    _Tp(0),
    highest,
    highest,
    _Tp(1),
    highest,
    _Tp(2),
  };

  constexpr std::array rhsValues{
    _Tp(7),
    highest,
    _Tp(0),
    _Tp(1),
    highest,
    _Tp(2),
    highest,
  };

  const auto lhs = makeSimdFromValues<TypeParam>(lhsValues);
  const auto rhs = makeSimdFromValues<TypeParam>(rhsValues);

  BOOST_TEST(
    to_array(saturating_mul(lhs, rhs)) ==
      applyBinaryToArray(lhs, rhs, expected_mul),
    boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto constexprLhs =
      makeSimdFromValues<TypeParam>(lhsValues);
    constexpr auto constexprRhs =
      makeSimdFromValues<TypeParam>(rhsValues);

    static_assert(
      to_array(saturating_mul(constexprLhs, constexprRhs)) ==
      applyBinaryToArray(constexprLhs, constexprRhs, expected_mul));
  }
#endif

  if constexpr (std::signed_integral<_Tp>) {
    // Exercise every sign combination, lower and upper saturation, both
    // operand orders, and multiplication of the lower endpoint by one.
    constexpr std::array signedLhsValues{
      _Tp(6),
      _Tp(-6),
      _Tp(-6),
      lowest,
      _Tp(2),
      highest,
      _Tp(-2),
      lowest,
      _Tp(-1),
      lowest,
      _Tp(1),
    };

    constexpr std::array signedRhsValues{
      _Tp(-7),
      _Tp(7),
      _Tp(-7),
      _Tp(2),
      lowest,
      _Tp(-2),
      highest,
      _Tp(-1),
      lowest,
      _Tp(1),
      lowest,
    };

    const auto signedLhs =
      makeSimdFromValues<TypeParam>(signedLhsValues);
    const auto signedRhs =
      makeSimdFromValues<TypeParam>(signedRhsValues);

    BOOST_TEST(
      to_array(saturating_mul(signedLhs, signedRhs)) ==
        applyBinaryToArray(signedLhs, signedRhs, expected_mul),
      boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto constexprLhs =
        makeSimdFromValues<TypeParam>(signedLhsValues);
      constexpr auto constexprRhs =
        makeSimdFromValues<TypeParam>(signedRhsValues);

      static_assert(
        to_array(saturating_mul(constexprLhs, constexprRhs)) ==
        applyBinaryToArray(
          constexprLhs, constexprRhs, expected_mul));
    }
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SatDiv, TypeParam, IntegerSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  constexpr auto lowest = std::numeric_limits<_Tp>::min();
  constexpr auto highest = std::numeric_limits<_Tp>::max();

  constexpr auto expected_div = [](_Tp lhs, _Tp rhs) {
    return xvec::simd::detail::saturating_div(lhs, rhs);
  };

  // Different elements exercise ordinary division, a zero numerator, and
  // division by one. No divisor is zero.
  constexpr std::array lhsValues{
    _Tp(42),
    _Tp(0),
    highest,
  };

  constexpr std::array rhsValues{
    _Tp(2),
    _Tp(1),
    _Tp(1),
  };

  const auto lhs = makeSimdFromValues<TypeParam>(lhsValues);
  const auto rhs = makeSimdFromValues<TypeParam>(rhsValues);

  BOOST_TEST(
    to_array(saturating_div(lhs, rhs)) ==
      applyBinaryToArray(lhs, rhs, expected_div),
    boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto constexprLhs =
      makeSimdFromValues<TypeParam>(lhsValues);
    constexpr auto constexprRhs =
      makeSimdFromValues<TypeParam>(rhsValues);

    static_assert(
      to_array(saturating_div(constexprLhs, constexprRhs)) ==
      applyBinaryToArray(constexprLhs, constexprRhs, expected_div));
  }
#endif

  if constexpr (std::signed_integral<_Tp>) {
    // Exercise all sign combinations and the only integral division that
    // requires saturation: lowest / -1.
    constexpr std::array signedLhsValues{
      lowest,
      _Tp(-42),
      _Tp(42),
      _Tp(-42),
      lowest,
    };

    constexpr std::array signedRhsValues{
      _Tp(1),
      _Tp(2),
      _Tp(-2),
      _Tp(-2),
      _Tp(-1),
    };

    const auto signedLhs =
      makeSimdFromValues<TypeParam>(signedLhsValues);
    const auto signedRhs =
      makeSimdFromValues<TypeParam>(signedRhsValues);

    BOOST_TEST(
      to_array(saturating_div(signedLhs, signedRhs)) ==
        applyBinaryToArray(signedLhs, signedRhs, expected_div),
      boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto constexprLhs =
        makeSimdFromValues<TypeParam>(signedLhsValues);
      constexpr auto constexprRhs =
        makeSimdFromValues<TypeParam>(signedRhsValues);

      static_assert(
        to_array(saturating_div(constexprLhs, constexprRhs)) ==
        applyBinaryToArray(
          constexprLhs, constexprRhs, expected_div));
    }
#endif
  }
}

template<typename To, typename V>
void checkSaturatingCast(const V& value)
{
  const auto expected_cast = [](auto element) {
    return xvec::simd::detail::saturating_cast<To>(element);
  };

  BOOST_TEST(
    to_array(saturating_cast<To>(value)) ==
      applyUnaryToArray(value, expected_cast),
    boost::test_tools::per_element());
}

template<typename V>
void checkAllSaturatingCasts(const V& value)
{
  checkSaturatingCast<std::uint8_t>(value);
  checkSaturatingCast<std::int8_t>(value);

  checkSaturatingCast<std::uint16_t>(value);
  checkSaturatingCast<std::int16_t>(value);

  checkSaturatingCast<std::uint32_t>(value);
  checkSaturatingCast<std::int32_t>(value);

  checkSaturatingCast<std::uint64_t>(value);
  checkSaturatingCast<std::int64_t>(value);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SatCast, TypeParam, IntegerSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  constexpr auto lowest = std::numeric_limits<_Tp>::min();
  constexpr auto highest = std::numeric_limits<_Tp>::max();

  // Exercise source endpoints, values adjacent to both endpoints, zero,
  // ordinary positive values, and a negative value for signed source types.
  constexpr std::array values{
    lowest,
    static_cast<_Tp>(lowest + _Tp(1)),
    std::signed_integral<_Tp> ? _Tp(-1) : _Tp(0),
    _Tp(0),
    _Tp(1),
    _Tp(42),
    static_cast<_Tp>(highest - _Tp(1)),
    highest,
  };

  const auto value = makeSimdFromValues<TypeParam>(values);
  checkAllSaturatingCasts(value);

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto constexprValue =
      makeSimdFromValues<TypeParam>(values);

    constexpr auto expected_uint8 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::uint8_t>(element);
    };
    constexpr auto expected_int8 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::int8_t>(element);
    };
    constexpr auto expected_uint16 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::uint16_t>(element);
    };
    constexpr auto expected_int16 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::int16_t>(element);
    };
    constexpr auto expected_uint32 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::uint32_t>(element);
    };
    constexpr auto expected_int32 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::int32_t>(element);
    };
    constexpr auto expected_uint64 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::uint64_t>(element);
    };
    constexpr auto expected_int64 = [](auto element) {
      return xvec::simd::detail::saturating_cast<std::int64_t>(element);
    };

    static_assert(
      to_array(saturating_cast<std::uint8_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_uint8));

    static_assert(
      to_array(saturating_cast<std::int8_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_int8));

    static_assert(
      to_array(saturating_cast<std::uint16_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_uint16));

    static_assert(
      to_array(saturating_cast<std::int16_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_int16));

    static_assert(
      to_array(saturating_cast<std::uint32_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_uint32));

    static_assert(
      to_array(saturating_cast<std::int32_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_int32));

    static_assert(
      to_array(saturating_cast<std::uint64_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_uint64));

    static_assert(
      to_array(saturating_cast<std::int64_t>(constexprValue)) ==
      applyUnaryToArray(constexprValue, expected_int64));
  }
#endif
}
