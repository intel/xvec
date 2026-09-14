//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include <xvec/detail/compat.hpp>

#include <concepts>
#include <cstdint>
#include <limits>

#if defined(_XVEC_HAS_CONSTEXPR)
#  define COMPAT_CONSTEXPR_CHECK(...) static_assert(__VA_ARGS__)
#else
#  define COMPAT_CONSTEXPR_CHECK(...)
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(AddSat, _Tp, IntegerTypes)
{
  constexpr _Tp lowest = std::numeric_limits<_Tp>::min();
  constexpr _Tp highest = std::numeric_limits<_Tp>::max();

  // Ordinary operations and exact boundaries.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(_Tp(20), _Tp(22)) == _Tp(42));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(_Tp(0), _Tp(0)) == _Tp(0));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(lowest, _Tp(0)) == lowest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(highest, _Tp(0)) == highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(_Tp(20), _Tp(22)), _Tp(42));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(_Tp(0), _Tp(0)), _Tp(0));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(lowest, _Tp(0)), lowest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(highest, _Tp(0)), highest);

  // Addition above the upper endpoint saturates for both signed and unsigned
  // integer types. Test both operand orders.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(highest, _Tp(1)) == highest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(_Tp(1), highest) == highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(highest, _Tp(1)), highest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(_Tp(1), highest), highest);

  if constexpr (std::signed_integral<_Tp>) {
    // Signed addition can also overflow below its lower endpoint.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(lowest, _Tp(-1)) == lowest);
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(_Tp(-1), lowest) == lowest);

    // Values adjacent to the boundaries must not be falsely saturated.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(highest, _Tp(-1)) == _Tp(highest - _Tp(1)));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(lowest, _Tp(1)) == _Tp(lowest + _Tp(1)));

    // Opposite-signed operands cannot overflow.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(_Tp(42), _Tp(-20)) == _Tp(22));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_add(_Tp(-20), _Tp(42)) == _Tp(22));

    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(lowest, _Tp(-1)), lowest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(_Tp(-1), lowest), lowest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(highest, _Tp(-1)), _Tp(highest - _Tp(1)));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(lowest, _Tp(1)), _Tp(lowest + _Tp(1)));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(_Tp(42), _Tp(-20)), _Tp(22));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_add(_Tp(-20), _Tp(42)), _Tp(22));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SubSat, _Tp, IntegerTypes)
{
  constexpr _Tp lowest = std::numeric_limits<_Tp>::min();
  constexpr _Tp highest = std::numeric_limits<_Tp>::max();

  // Ordinary operations and exact boundaries.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(_Tp(42), _Tp(20)) == _Tp(22));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(_Tp(42), _Tp(42)) == _Tp(0));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(lowest, _Tp(0)) == lowest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(highest, _Tp(0)) == highest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(highest, highest) == _Tp(0));

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(_Tp(42), _Tp(20)), _Tp(22));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(_Tp(42), _Tp(42)), _Tp(0));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(lowest, _Tp(0)), lowest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(highest, _Tp(0)), highest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(highest, highest), _Tp(0));

  // Subtracting one from the lowest value saturates for both signed and
  // unsigned integer types.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(lowest, _Tp(1)) == lowest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(lowest, _Tp(1)), lowest);

  if constexpr (std::unsigned_integral<_Tp>) {
    // Unsigned subtraction saturates at zero rather than wrapping.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(_Tp(0), _Tp(1)) == _Tp(0));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(_Tp(1), _Tp(1)) == _Tp(0));

    // A representable result adjacent to the upper endpoint is unchanged.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(highest, _Tp(1)) == _Tp(highest - _Tp(1)));

    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(_Tp(0), _Tp(1)), _Tp(0));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(_Tp(1), _Tp(1)), _Tp(0));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(highest, _Tp(1)), _Tp(highest - _Tp(1)));
  } else {
    // Subtracting a negative operand can overflow toward the upper endpoint.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(highest, _Tp(-1)) == highest);

    // Values adjacent to the boundaries must remain unsaturated.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(lowest, _Tp(-1)) == _Tp(lowest + _Tp(1)));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(highest, _Tp(1)) == _Tp(highest - _Tp(1)));

    // Ordinary subtraction involving negative values.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(_Tp(20), _Tp(-22)) == _Tp(42));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_sub(_Tp(-20), _Tp(22)) == _Tp(-42));

    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(highest, _Tp(-1)), highest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(lowest, _Tp(-1)), _Tp(lowest + _Tp(1)));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(highest, _Tp(1)), _Tp(highest - _Tp(1)));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(_Tp(20), _Tp(-22)), _Tp(42));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_sub(_Tp(-20), _Tp(22)), _Tp(-42));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MulSat, _Tp, IntegerTypes)
{
  constexpr _Tp lowest = std::numeric_limits<_Tp>::min();
  constexpr _Tp highest = std::numeric_limits<_Tp>::max();

  // Ordinary multiplication, zero, identity, and upper saturation.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(6), _Tp(7)) == _Tp(42));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(0), highest) == _Tp(0));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(highest, _Tp(0)) == _Tp(0));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(highest, _Tp(1)) == highest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(1), highest) == highest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(highest, _Tp(2)) == highest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(2), highest) == highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(6), _Tp(7)), _Tp(42));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(0), highest), _Tp(0));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(highest, _Tp(0)), _Tp(0));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(highest, _Tp(1)), highest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(1), highest), highest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(highest, _Tp(2)), highest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(2), highest), highest);

  if constexpr (std::signed_integral<_Tp>) {
    // Exercise all four operand-sign combinations without overflow.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(6), _Tp(-7)) == _Tp(-42));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(-6), _Tp(7)) == _Tp(-42));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(-6), _Tp(-7)) == _Tp(42));

    // Lower-endpoint overflow with both operand orders.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(lowest, _Tp(2)) == lowest);
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(2), lowest) == lowest);
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(highest, _Tp(-2)) == lowest);
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(-2), highest) == lowest);

    // Negating the lowest value mathematically produces one more than the
    // upper endpoint, so the result must saturate high.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(lowest, _Tp(-1)) == highest);
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(-1), lowest) == highest);

    // Multiplication by positive one preserves the lower endpoint.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(lowest, _Tp(1)) == lowest);
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_mul(_Tp(1), lowest) == lowest);

    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(6), _Tp(-7)), _Tp(-42));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(-6), _Tp(7)), _Tp(-42));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(-6), _Tp(-7)), _Tp(42));

    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(lowest, _Tp(2)), lowest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(2), lowest), lowest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(highest, _Tp(-2)), lowest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(-2), highest), lowest);

    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(lowest, _Tp(-1)), highest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(-1), lowest), highest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(lowest, _Tp(1)), lowest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_mul(_Tp(1), lowest), lowest);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(DivSat, _Tp, IntegerTypes)
{
  constexpr _Tp lowest = std::numeric_limits<_Tp>::min();
  constexpr _Tp highest = std::numeric_limits<_Tp>::max();

  // Ordinary division and identity cases.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(_Tp(42), _Tp(2)) == _Tp(21));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(_Tp(0), _Tp(1)) == _Tp(0));
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(highest, _Tp(1)) == highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(_Tp(42), _Tp(2)), _Tp(21));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(_Tp(0), _Tp(1)), _Tp(0));
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(highest, _Tp(1)), highest);

  if constexpr (std::signed_integral<_Tp>) {
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(lowest, _Tp(1)) == lowest);
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(_Tp(-42), _Tp(2)) == _Tp(-21));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(_Tp(42), _Tp(-2)) == _Tp(-21));
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(_Tp(-42), _Tp(-2)) == _Tp(21));

    // This is the only nonzero integral division that requires saturation.
    COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_div(lowest, _Tp(-1)) == highest);

    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(lowest, _Tp(1)), lowest);
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(_Tp(-42), _Tp(2)), _Tp(-21));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(_Tp(42), _Tp(-2)), _Tp(-21));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(_Tp(-42), _Tp(-2)), _Tp(21));
    BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_div(lowest, _Tp(-1)), highest);
  }
}

BOOST_AUTO_TEST_CASE(SaturateCast)
{
  constexpr auto int8Lowest = std::numeric_limits<int8_t>::min();
  constexpr auto int8Highest = std::numeric_limits<int8_t>::max();
  constexpr auto uint8Highest = std::numeric_limits<uint8_t>::max();

  // Same-type conversion.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(int8_t{42}) == int8_t{42});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(int8_t{42}), int8_t{42});

  // Values exactly at destination endpoints remain unchanged.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(-128) == int8Lowest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(127) == int8Highest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint8_t>(0) == uint8_t{0});
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint8_t>(255) == uint8Highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(-128), int8Lowest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(127), int8Highest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint8_t>(0), uint8_t{0});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint8_t>(255), uint8Highest);

  // Signed-to-signed narrowing, including both saturation directions.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(-129) == int8Lowest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(128) == int8Highest);
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(42) == int8_t{42});

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(-129), int8Lowest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(128), int8Highest);
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(42), int8_t{42});

  // Unsigned-to-signed narrowing.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(uint64_t{42}) == int8_t{42});
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int8_t>(uint64_t{255}) == int8Highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(uint64_t{42}), int8_t{42});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int8_t>(uint64_t{255}), int8Highest);

  // Signed-to-unsigned narrowing, including both saturation directions.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint8_t>(-1) == uint8_t{0});
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint8_t>(42) == uint8_t{42});
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint8_t>(256) == uint8Highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint8_t>(-1), uint8_t{0});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint8_t>(42), uint8_t{42});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint8_t>(256), uint8Highest);

  // Unsigned-to-unsigned narrowing.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint8_t>(uint64_t{42}) == uint8_t{42});
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint8_t>(uint64_t{256}) == uint8Highest);

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint8_t>(uint64_t{42}), uint8_t{42});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint8_t>(uint64_t{256}), uint8Highest);

  // Widening conversions preserve their values.
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int64_t>(int8_t{-42}) == int64_t{-42});
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<uint64_t>(uint8_t{42}) == uint64_t{42});
  COMPAT_CONSTEXPR_CHECK(xvec::simd::detail::saturating_cast<int64_t>(uint8_t{255}) == int64_t{255});

  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int64_t>(int8_t{-42}), int64_t{-42});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<uint64_t>(uint8_t{42}), uint64_t{42});
  BOOST_CHECK_EQUAL(xvec::simd::detail::saturating_cast<int64_t>(uint8_t{255}), int64_t{255});
}

#undef COMPAT_CONSTEXPR_CHECK
