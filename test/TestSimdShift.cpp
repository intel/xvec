//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

#include "TypesToTest.hpp"

template<typename T> class SimdShiftFixture {
public:
  // Create a vector of shift values, up to the maximum number of bits in each element.
  static constexpr auto numBits = sizeof(typename T::value_type) * 8;

  /// Get a vector of random shift values. They must be positive values.
  using UNSIGNED_ELEMENT_TYPE = std::make_unsigned_t<typename T::value_type>;
  using UNSIGNED_T = xvec::simd::vec<UNSIGNED_ELEMENT_TYPE, T::size()>;
  const T shifts = T(GetRandomVector<UNSIGNED_T>(numBits - 1));

  const T v0 = GetRandomVector<T>(32);

  /// Get a single shift value for used in scalar shifting.
  const typename T::value_type scalarShift = getMedian(shifts);
};

BOOST_AUTO_TEST_CASE_TEMPLATE(ShiftLeftSimd, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;
  using tvt = typename TypeParam::value_type;
  const auto expected = applyBinary(f.v0, f.shifts, [=](tvt l, tvt r) { return tvt(l << r); });
  BOOST_SIMD_EQUAL(f.v0 << f.shifts, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ShiftRightSimd, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;
  using tvt = typename TypeParam::value_type;
  const auto expected = applyBinary(f.v0, f.shifts, [=](tvt l, tvt r) { return tvt(l >> r); });
  BOOST_SIMD_EQUAL(f.v0 >> f.shifts, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ShiftLeftScalar, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;
  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [f](tvt l) { return tvt(l << f.scalarShift); });
  BOOST_SIMD_EQUAL(f.v0 << f.scalarShift, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ShiftRightScalar, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;
  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary(f.v0, [f](tvt l) { return tvt(l >> f.scalarShift); });
  BOOST_SIMD_EQUAL(f.v0 >> f.scalarShift, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AssignmentShiftLeftSimd, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyBinary(f.v0, f.shifts, [=](tvt l, tvt r) { return tvt(l << r); });

  auto assignedValue = f.v0;
  assignedValue <<= f.shifts;

  BOOST_SIMD_EQUAL(assignedValue, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AssignmentShiftRightSimd, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyBinary(f.v0, f.shifts, [=](tvt l, tvt r) { return tvt(l >> r); });

  auto assignedValue = f.v0;
  assignedValue >>= f.shifts;

  BOOST_SIMD_EQUAL(assignedValue, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AssignmentShiftLeftScalar, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;
  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary (f.v0, [f](tvt v) { return tvt(v << f.scalarShift); });

  auto assignedValue = f.v0;
  assignedValue <<= f.scalarShift;

  BOOST_SIMD_EQUAL(assignedValue, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AssignmentShiftRightScalar, TypeParam, IntegerSimdTypes)
{
  SimdShiftFixture<TypeParam> f;

  using tvt = typename TypeParam::value_type;
  const auto expected = applyUnary (f.v0, [f](tvt v) { return tvt(v >> f.scalarShift); });

  auto assignedValue = f.v0;
  assignedValue >>= f.scalarShift;

  BOOST_SIMD_EQUAL(assignedValue, expected);
}
