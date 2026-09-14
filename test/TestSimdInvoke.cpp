//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

using xvec::simd::basic_vec;
using xvec::simd::basic_mask;
using xvec::simd::vec;
using xvec::simd::mask;
using xvec::simd::vec_or_mask_type;

using std::tuple;

void CheckSimdOrMaskType()
{
  static_assert(vec_or_mask_type<vec<float>>);
  static_assert(vec_or_mask_type<mask<float>>);

  static_assert(vec_or_mask_type<const vec<float>>);
  static_assert(vec_or_mask_type<volatile vec<float>>);

  static_assert(!vec_or_mask_type<float>);
}

void CheckAllSizesAreEqual()
{
  using xvec::simd::detail::are_all_sizes_equal;
  static_assert(are_all_sizes_equal<vec<float, 13>::size(), mask<char, 13>::size(), vec<double, 13>::size()>);
  static_assert(!are_all_sizes_equal<vec<float, 12>::size(), mask<char, 13>::size(), vec<double, 13>::size()>);
}

xvec::simd::detail::target_overloads CheckEqual = {
  []<typename T, typename ABI>(basic_vec<T, ABI> computed, basic_vec<T, ABI> expected) { BOOST_SIMD_EQUAL(computed, expected); },
  []<std::size_t _Bytes, typename ABI>(basic_mask<_Bytes, ABI> computed, basic_mask<_Bytes, ABI> expected) { BOOST_TEST(computed.to_bitset() == expected.to_bitset()); }
};

/// Check calling a simdable once.
static void CheckCallSimdOrMaskOnce(const vec_or_mask_type auto& input)
{
  constexpr int size = decltype(input.size)::value;

  // Non-indexed, no return
  auto fnWithoutIndexNoReturn = [=](auto s)
  {
    CheckEqual(s, input);
  };
  xvec::simd::chunked_invoke<size>(fnWithoutIndexNoReturn, input);

  // Indexed, no return
  auto fnWithIndexNoReturn = [=](auto s, std::size_t idx)
  {
    CheckEqual(s, input);
    BOOST_TEST(idx == 0);
  };
  xvec::simd::chunked_invoke<size>(fnWithIndexNoReturn, input);

  // Non-indexed, with return
  auto fnWithoutIndexWithReturn = [=](auto s)
  {
    CheckEqual(s, input);
    return s;
  };
  auto withoutIndexComputed = xvec::simd::chunked_invoke<size>(fnWithoutIndexWithReturn, input);
  CheckEqual(withoutIndexComputed, input);

  // Indexed, with return
  auto fnWithIndexWithReturn = [=](auto s, std::size_t idx)
  {
    CheckEqual(s, input);
    BOOST_TEST(idx == 0);
    return s;
  };
  auto withIndexComputed = xvec::simd::chunked_invoke<size>(fnWithIndexWithReturn, input);
  CheckEqual(withIndexComputed, input);
}

/// Check calling a simdable in multiples of 3.
template<typename TypeParam>
static void CheckCallSimdOrMaskInMultiplesOf3()
{
  // For all the tests below we need to ensure that every element of the input
  // is processed, so keep track of those elements by passing in iota, and then
  // using an ordered set to track which bits have been processed. After chunked_invoke
  // has been called, every element should have been written to. This is
  // necessary because chunked_invoke does not necessarily execute each invocation
  // of the sub-simd in order.
  auto checkBit = [](auto& bits, auto s) {
    for (int i=0; i<s.size; ++i)
    {
      BOOST_TEST(bits[s[i]] == false); // Can't have been processed already.
      bits[s[i]] = true;                  // Now mark it as processed in the expectation it will never be seen again.
    }
  };

  // Non-indexed, no return
  {
    std::bitset<TypeParam::size> ninr;
    auto fnWithoutIndexNoReturn = [&](auto s) { checkBit(ninr, s); };
    xvec::simd::chunked_invoke<3>(fnWithoutIndexNoReturn, xvec::simd::iota<TypeParam>);
    BOOST_TEST(ninr.all() == true);
  }

  // Indexed, no return
  {
    std::bitset<TypeParam::size> ninr;
    std::bitset<TypeParam::size> indexes;
    auto fnWithIndexNoReturn = [&](auto s, std::size_t i) {
      checkBit(ninr, s); 
      BOOST_TEST(indexes[i] == false); // Can't have called this index already.
      BOOST_TEST(i % 3 == 0);          // Must index in a correct multiple.
      indexes[i] = true;
    };
    xvec::simd::chunked_invoke<3>(fnWithIndexNoReturn, xvec::simd::iota<TypeParam>);
    BOOST_TEST(ninr.all() == true);
    BOOST_TEST(indexes.count() == (TypeParam::size + 2) / 3); // Correct number of invocations must have happened.
  }

  // Non-indexed, with return
  {
    std::bitset<TypeParam::size> niwr;

    auto fnWithoutIndexWithReturn = [&](auto s) { checkBit(niwr, s); return s; };
    auto withoutIndexComputed = xvec::simd::chunked_invoke<3>(fnWithoutIndexWithReturn, xvec::simd::iota<TypeParam>);
    BOOST_TEST(niwr.all() == true);
    CheckEqual(withoutIndexComputed, xvec::simd::iota<TypeParam>);
  }

  // Indexed, with no return
  {
    std::bitset<TypeParam::size> iwr;
    std::bitset<TypeParam::size> indexes;
    auto fnWithIndexWithReturn = [&](auto s, auto i) {
      checkBit(iwr, s);
      BOOST_TEST(indexes[i] == false); // Can't have called this index already.
      BOOST_TEST(i % 3 == 0);          // Must index in a correct multiple.
      indexes[i] = true;
      return s;
    };
    auto withIndexComputed = xvec::simd::chunked_invoke<3>(fnWithIndexWithReturn, xvec::simd::iota<TypeParam>);
    BOOST_TEST(iwr.all() == true);
    BOOST_TEST(indexes.count() == (TypeParam::size + 2) / 3); // Correct number of invocations must have happened.
    CheckEqual(withIndexComputed, xvec::simd::iota<TypeParam>);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ApplyOnceToSimd, TypeParam, AllSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  CheckCallSimdOrMaskOnce(f.v0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ApplyInMultiplesOf3ToSimd, TypeParam, ArithmeticSimdTypes)
{
  CheckCallSimdOrMaskInMultiplesOf3<TypeParam>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ApplyOnceToSimdMask, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> f;
  CheckCallSimdOrMaskOnce(f.mask0);
}
