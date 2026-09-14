//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define XVEC_ALWAYS_RANGE_CHECK_MEMORY

#include "SimdTestUtilities.hpp"
#include "TypesToTest.hpp"

// :TODO: The tests in this file assume that the values are loaded from real-valued, so
// what should happen when a complex value is loaded from a real-valued source.
// Should it convert, or break? For the moment it breaks.
using LoadableSimdTypes = ArithmeticSimdTypes;

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckExceptionOnLoadStoreRangeError, TypeParam, LoadableSimdTypes)
{
  // Very small range to use for checking. Part of a bigger span to prevent
  // segfault if this doesn't work.
  std::array<typename TypeParam::value_type, 10> r;
  auto smallRange = std::span(r.begin(), 1);

  if constexpr (TypeParam::size > 1)
  {
    BOOST_CHECK_THROW((void)xvec::simd::unchecked_load<TypeParam>(smallRange), std::out_of_range);
    BOOST_CHECK_THROW((void)xvec::simd::unchecked_store(xvec::simd::iota<TypeParam>, smallRange), std::out_of_range);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckExceptionOnMaskedLoadStoreRangeError, TypeParam, LoadableSimdTypes)
{
  // Very small range to use for checking. Part of a bigger span to prevent
  // segfault if this doesn't work.
  std::array<typename TypeParam::value_type, 10> r;
  auto smallRange = std::span(r.begin(), 2);

  // Note that at least 2 elements are needed because this is a masked
  // operation, so there must be at least one unmasked and one masked operation.
  if constexpr (TypeParam::size > 2)
  {
    auto m = typename TypeParam::mask_type([](auto idx) { return idx % 2 == 0; });

    BOOST_CHECK_THROW((void)xvec::simd::unchecked_load<TypeParam>(smallRange, m), std::out_of_range);
    BOOST_CHECK_THROW((void)xvec::simd::unchecked_store(xvec::simd::iota<TypeParam>, smallRange, m), std::out_of_range);

    // Check that masked load/store don't throw an exception if the mask itself is used to limit their scope.
    auto scopeMask = xvec::simd::mask_from_count<TypeParam>(2);
    xvec::simd::unchecked_load<TypeParam>(smallRange, scopeMask);
    xvec::simd::unchecked_store(xvec::simd::iota<TypeParam>, smallRange, scopeMask);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckExceptionOnGatherScatterRangeError, TypeParam, LoadableSimdTypes)
{
  // Very small range to use for checking. Part of a bigger span to prevent
  // segfault if this doesn't work.
  std::array<typename TypeParam::value_type, 10> r;
  auto smallRange = std::span(r.begin(), 2);

  xvec::simd::vec<int, TypeParam::size> invalidIndexes(5);
  BOOST_CHECK_THROW((void)xvec::simd::unchecked_gather_from(smallRange, invalidIndexes), std::out_of_range);
  BOOST_CHECK_THROW((void)xvec::simd::unchecked_scatter_to(xvec::simd::iota<TypeParam>, smallRange, invalidIndexes), std::out_of_range);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckExceptionOnMaskedGatherScatterRangeError, TypeParam, LoadableSimdTypes)
{
  // Very small range to use for checking. Part of a bigger span to prevent
  // segfault if this doesn't work.
  std::array<typename TypeParam::value_type, 10> r;
  auto smallRange = std::span(r.begin(), 2);

  auto m = xvec::simd::mask<int, TypeParam::size>([](auto idx) { return idx % 2 == 0; });

  xvec::simd::vec<int, TypeParam::size> invalidIndexes(5);

  BOOST_CHECK_THROW((void)xvec::simd::unchecked_gather_from(smallRange, m, invalidIndexes), std::out_of_range);
  BOOST_CHECK_THROW((void)xvec::simd::unchecked_scatter_to(xvec::simd::iota<TypeParam>, smallRange, m, invalidIndexes), std::out_of_range);

  // Create a new set of indexes where all the masked indexes are valid and the
  // unmasked indexes are not valid. This ensures that the gather_from and
  // scatter_to are only checking the indexes which would be used.
  auto iotaIndexes = xvec::simd::iota<xvec::simd::vec<int, TypeParam::size>>;
  auto invalidIndexMask = iotaIndexes < 2;
  unchecked_gather_from(smallRange, invalidIndexMask, iotaIndexes); // Should run without exception
  unchecked_scatter_to(xvec::simd::iota<TypeParam>, smallRange, invalidIndexMask, iotaIndexes); // Should run without exception

}
