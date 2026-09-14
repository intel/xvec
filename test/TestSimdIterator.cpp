//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <xvec/simd>

// make iterators printable for the test results.
template<typename _Simd>
std::ostream& operator<<(std::ostream& stream, const xvec::simd::simd_iterator<_Simd>& iter) {
  stream << "simd_iterator<" << (void*)iter.data << ", " << iter.index << ">";
  return stream;
}

#include "SimdTestUtilities.hpp"
#include "TypesToTest.hpp"

// Test all mask types, and a selection of simd types of different types. No
// need to test every simd type since the iterators treat their contents as
// indexable containers, with no care about the element type.
using AllIterableSimdTypes = boost::mp11::mp_append<UnsignedSimdTypes, AllSimdMaskTypes>;

template<xvec::simd::vec_or_mask_type _S>
struct IterableTestFixture {
  static constexpr typename _S::value_type limit = std::numeric_limits<typename _S::value_type>::max();

  IterableTestFixture() {
    if constexpr (std::same_as<typename _S::value_type, bool>) {
      v0 = _S(getRandomBitset<_S::size>());
      v1 = _S(getRandomBitset<_S::size>());
    } else {
      v0 = GetRandomVector<_S>(limit);
      v1 = GetRandomVector<_S>(limit);
    }
  }

  _S v0, v1;
};

BOOST_AUTO_TEST_CASE_TEMPLATE(IteratorComparisonsForRangeBounds, TypeParam, AllIterableSimdTypes)
{
  IterableTestFixture<TypeParam> f;

  BOOST_TEST( (f.v0.begin() == f.v0.begin()));
  BOOST_TEST(!(f.v0.begin() != f.v0.begin()));
  BOOST_TEST(!(f.v0.begin() <  f.v0.begin()));
  BOOST_TEST(!(f.v0.begin() >  f.v0.begin()));
  BOOST_TEST( (f.v0.begin() <= f.v0.begin()));
  BOOST_TEST( (f.v0.begin() >= f.v0.begin()));

  // iterator/sentinel comparisons supported by working draft: equality/inequality
  BOOST_TEST(!(f.v0.begin() == f.v0.end()));
  BOOST_TEST( (f.v0.begin() != f.v0.end()));

  // sized sentinel distance checks
  BOOST_TEST((f.v0.end() - f.v0.begin()) == static_cast<int>(TypeParam::size));
  BOOST_TEST((f.v0.begin() - f.v0.end()) == -static_cast<int>(TypeParam::size));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IteratorComparisonsForArbitraryiters, TypeParam, AllIterableSimdTypes)
{
  IterableTestFixture<TypeParam> f;

  // Assume at least 3 iterator positions are available.
  if (TypeParam::size < 3) return;

  auto p0 = xvec::simd::simd_iterator(f.v0, 0);
  auto p1 = xvec::simd::simd_iterator(f.v0, 1);

  BOOST_TEST(p0 < p1);
  BOOST_TEST(p0 <= p1);
  BOOST_TEST(!(p0 > p1));
  BOOST_TEST(!(p0 >= p1));
  BOOST_TEST(!(p1 <  p0));
  BOOST_TEST(!(p1 <= p0));
  BOOST_TEST(p1 > p0);
  BOOST_TEST(p1 >= p0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IteratorIncrement, TypeParam, AllIterableSimdTypes)
{
  IterableTestFixture<TypeParam> f;

  // Assume at least 3 iterator positions are available.
  if (TypeParam::size < 3) return;

  auto p0 = xvec::simd::simd_iterator(f.v0, 0);
  auto p1 = xvec::simd::simd_iterator(f.v0, 1);
  auto p2 = xvec::simd::simd_iterator(f.v0, 2);
  auto pfinal = xvec::simd::simd_iterator(f.v0, TypeParam::size - 1);

  auto t0 = p0;
  BOOST_TEST((++t0) == p1);
  BOOST_TEST((p0 + 1) == p1);
  BOOST_TEST((p0 + 2) == p2);
  BOOST_TEST((1 + p0) == p1);
  BOOST_TEST((2 + p0) == p2);

  BOOST_TEST((p2 + (-1)) == p1);
  BOOST_TEST((p2 + (-2)) == p0);
  BOOST_TEST((-1 + p2) == p1);
  BOOST_TEST((-2 + p2) == p0);

  // end() is a sentinel, so validate relative position via distance
  BOOST_TEST((f.v0.end() - pfinal) == static_cast<int>(1));

  auto t1 = p0;
  t1 += 2;
  BOOST_TEST(t1 == p2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IteratorDecrement, TypeParam, AllIterableSimdTypes)
{
  IterableTestFixture<TypeParam> f;

  // Assume at least 3 iterator positions are available.
  if (TypeParam::size < 3) return;

  auto p0 = xvec::simd::simd_iterator(f.v0, 0);
  auto p1 = xvec::simd::simd_iterator(f.v0, 1);
  auto p2 = xvec::simd::simd_iterator(f.v0, 2);
  auto pfinal = xvec::simd::simd_iterator(f.v0, TypeParam::size - 1);

  BOOST_TEST(p2 == p2);
  auto t0 = p2;
  BOOST_TEST((--t0) == p1);
  BOOST_TEST((p2 - 1) == p1);
  BOOST_TEST((p2 - 2) == p0);

  // end() is a sentinel so validate relative position via distance
  BOOST_TEST((f.v0.end() - pfinal) == static_cast<int>(1));

  auto t1 = p2;
  t1 -= 2;
  BOOST_TEST(t1 == p0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IteratorDereference, TypeParam, AllIterableSimdTypes)
{
  IterableTestFixture<TypeParam> f;

  // Assume at least 3 iterator positions are available.
  if (TypeParam::size < 3) return;

  auto p0 = xvec::simd::simd_iterator(f.v0, 0);
  auto p1 = xvec::simd::simd_iterator(f.v0, 1);
  auto p2 = xvec::simd::simd_iterator(f.v0, 2);
  auto pfinal = xvec::simd::simd_iterator(f.v0, TypeParam::size - 1);

  BOOST_TEST(*p0 == f.v0[0]);
  BOOST_TEST(*p1 == f.v0[1]);
  BOOST_TEST(*p2 == f.v0[2]);

  BOOST_TEST(p0[0] == f.v0[0]);
  BOOST_TEST(p0[1] == f.v0[1]);
  BOOST_TEST(p0[2] == f.v0[2]);

  BOOST_TEST(p2[-1] == f.v0[1]);
  BOOST_TEST(p2[-2] == f.v0[0]);

  // end() is a sentinel, so use the last valid iterator for dereference checks
  BOOST_TEST(*pfinal == f.v0[TypeParam::size - 1]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IteratorRange, TypeParam, AllIterableSimdTypes)
{
  IterableTestFixture<TypeParam> f;

  // Copy
  std::vector<typename TypeParam::value_type> v0;
  for (auto e : f.v0) v0.push_back(e);
  BOOST_TEST(v0 == f.v0, boost::test_tools::per_element());

  // Reverse
  std::vector<typename TypeParam::value_type> v1;
  for (auto e : f.v0 | std::views::reverse) v1.insert(v1.begin(), e);
  BOOST_TEST(v1 == f.v0, boost::test_tools::per_element());

  // Doesn't need to be exhaustive - a couple of examples checking basic behaviour is okay.
}
