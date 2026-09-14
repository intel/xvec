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

BOOST_AUTO_TEST_CASE_TEMPLATE(Usable, TypeParam, AllSimdTypes)
{
  static_assert(std::is_default_constructible_v<TypeParam>);
  static_assert(std::is_destructible_v<TypeParam>);
  static_assert(std::is_default_constructible_v<typename TypeParam::mask_type>);
  static_assert(std::is_destructible_v<typename TypeParam::mask_type>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IsSimd, TypeParam, AllSimdTypes)
{
  // Or std::simd, or x86::simd.
  BOOST_TEST(xvec::simd::detail::is_vec_v<TypeParam> == true);
  BOOST_TEST(xvec::simd::detail::is_vec_v<typename TypeParam::value_type> == false);
  //static_assert(simd::is_abi_tag_v<typename TypeParam::abi_type>); // :TODO: Implement

  // Various other types aren't vectors.
  BOOST_TEST(xvec::simd::detail::is_vec_v<std::string> == false);
  BOOST_TEST(xvec::simd::detail::is_vec_v<double> == false);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AssignScalar, TypeParam, AllSimdTypes)
{
  const auto value = SimdTestFixture<TypeParam>().getRandomValue();

  TypeParam simd;
  simd = TypeParam(value);

  BOOST_SIMD_EQUAL(simd, TypeParam(value));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReadAccessByIndex, TypeParam, AllSimdTypes)
{
  alignas(64) const auto elements = make_iota<TypeParam>(3);

  const auto constData = TypeParam(elements);

  // Don't replace with a container test. It really is testing the [] operator.
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    BOOST_CHECK(constData[i] == elements[i]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprReadAccessByIndex, TypeParam, AllSimdTypes)
{
  constexpr auto elements = make_iota<TypeParam>();
  constexpr auto v = TypeParam(elements);

  BOOST_CHECK(v[0] == elements[0]);

  if (TypeParam::size > 4)
    BOOST_CHECK(v[3] == elements[3]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReadAccessByIndexOnNonConst, TypeParam, AllSimdTypes)
{
  alignas(64) const auto elements = make_iota<TypeParam>(3);

  auto data = TypeParam(elements);

  // Don't replace with an BOOST_TEST. It really is testing the [] operator.
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    BOOST_CHECK(data[i] == elements[i]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(StringOutputOfDuplicatedValue, TypeParam, AllSimdTypes)
{
  const auto arbitraryValue = SimdTestFixture<TypeParam>().getRandomValue();

  TypeParam dup(arbitraryValue);

  // Get the string representation of the element value.
  std::ostringstream sv;
  sv << arbitraryValue;

  std::ostringstream s;
  s << dup;

  // More about the formatting than the actual value.
  std::string expected;
  for (std::size_t i = 0; i < TypeParam::size(); ++i)
  {
    expected += '[' + std::to_string((TypeParam::size() - 1) - i) + "]:" + sv.str();
    if (i != (TypeParam::size() - 1))
      expected += ' ';
  }

  BOOST_TEST(s.str() == expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SelectSimd, TypeParam, AllSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // Essentially random selection bits.
  constexpr std::bitset<TypeParam::size> mask_bitset = 0x3681d8a2;
  constexpr typename TypeParam::mask_type mask(mask_bitset);

  typename SimdTestFixture<TypeParam>::test_array_type expected = {};
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = mask_bitset.test(i) ? f.v0[i] : f.v1[i];

  BOOST_TEST(to_array(select(mask, f.v0, f.v1)) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprSelectSimd, TypeParam, ArithmeticSimdTypes)
{
  // Essentially random selection bits.
  constexpr std::bitset<TypeParam::size> mask_bitset = 0x3681d8a2;

  constexpr TypeParam v0 = typename TypeParam::value_type(12);
  constexpr TypeParam v1 = typename TypeParam::value_type(23);

  typename SimdTestFixture<TypeParam>::test_array_type expected = {};
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = mask_bitset.test(i) ? v0[i] : v1[i];

  #if defined(_XVEC_HAS_CONSTEXPR)
    constexpr typename TypeParam::mask_type mask(mask_bitset);
    constexpr auto computed = select(mask, v0, v1);
    BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());
  #endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SelectBool, TypeParam, AllSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  const auto v0 = f.v0[0];
  const auto v1 = f.v1[0];

  BOOST_TEST(xvec::simd::select(false, v0, v1) == v1);
  BOOST_TEST(xvec::simd::select(true , v0, v1) == v0);
}

BOOST_AUTO_TEST_CASE(RebindVecChangesElementType)
{
  using Source = xvec::simd::vec<short, 8>;
  using Expected = xvec::simd::vec<int, 8>;

  constexpr auto source = xvec::simd::iota<Source>;
  auto computed = xvec::rebind_cast<int>(source);

  static_assert(std::same_as<decltype(computed), Expected>);
  BOOST_SIMD_EQUAL(computed, Expected(make_iota<Expected>()));
}

BOOST_AUTO_TEST_CASE(RebindMaskChangesElementType)
{
  using Source = xvec::simd::mask<short, 8>;
  using Rebound = xvec::simd::rebind_t<int, Source>;
  using Expected = xvec::simd::mask<int, 8>;

  static_assert(std::same_as<Rebound, Expected>);
  static_assert(Rebound::size == Source::size);
  static_assert(xvec::simd::simd_mask_element_size_v<Rebound> == sizeof(int));

  constexpr Source source([](std::size_t i) { return i % 2 == 0; });
  const Rebound rebound(source);
  BOOST_TEST(rebound.to_ullong() == source.to_ullong()); // bitset would be better, but isn't constexpr.

#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr Rebound rebound_ce(source);
  static_assert(rebound_ce.to_ullong() == source.to_ullong());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SimdBitCast, TypeParam, AllSimdTypes)
{
  // There are limits on total numbers of elements in some compilers, so only
  // test those values which don't overflow that size when converted to bytes.
  using _E = typename TypeParam::value_type;
  if constexpr ((sizeof(_E) * TypeParam::size) <= xvec::simd::vec<_E>::size())
  {
    SimdTestFixture<TypeParam> f;

    constexpr auto elementByteSize = sizeof(typename TypeParam::value_type);
    constexpr auto simdByteSize = elementByteSize * TypeParam::size();

    // Convert to its plain byte representation. Absolutely any valid simd type should be allowed such a conversion.
    const auto computed = simd_bit_cast<uint8_t>(f.v0);
    BOOST_TEST(computed.size() == simdByteSize);

    // The internal representations should be identical.
    std::array<uint8_t, simdByteSize> originalData;
    std::copy_n((const uint8_t *)&f.v0, simdByteSize, originalData.begin());
    std::array<uint8_t, simdByteSize> computedData;
    partial_store(computed, computedData);
    BOOST_TEST(computedData == originalData);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SimdBitCastByVecType, TypeParam, AllSimdTypes)
{
  // There are limits on total numbers of elements in some compilers, so only
  // test those values which don't overflow that size when converted to bytes.
  using _E = typename TypeParam::value_type;
  if constexpr ((sizeof(_E) * TypeParam::size) <= xvec::simd::vec<_E>::size())
  {
    SimdTestFixture<TypeParam> f;

    constexpr auto elementByteSize = sizeof(typename TypeParam::value_type);
    constexpr auto simdByteSize = elementByteSize * TypeParam::size();

    // Convert to an explicit new vector type.
    using ByteVecType = xvec::simd::vec<uint8_t, simdByteSize>;

    // Convert to its plain byte representation. Absolutely any valid simd type should be allowed such a conversion.
    const auto computed = simd_bit_cast<ByteVecType>(f.v0);
    BOOST_TEST(computed.size() == simdByteSize);

    // The internal representations should be identical.
    std::array<uint8_t, simdByteSize> originalData;
    std::copy_n((const uint8_t *)&f.v0, simdByteSize, originalData.begin());
    std::array<uint8_t, simdByteSize> computedData;
    partial_store(computed, computedData);
    BOOST_TEST(computedData == originalData);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SimdBitCastConstexpr, TypeParam, AllSimdTypes)
{
  // There are limits on total numbers of elements in some compilers, so only
  // test those values which don't overflow that size when converted to bytes.
  using _E = typename TypeParam::value_type;
  if constexpr ((sizeof(_E) * TypeParam::size) <= xvec::simd::vec<_E>::size())
  {
    constexpr auto elementByteSize = sizeof(typename TypeParam::value_type);
    constexpr auto simdByteSize = elementByteSize * TypeParam::size();

    // Convert to its plain byte representation. Absolutely any valid simd type should be allowed such a conversion.
    constexpr auto original = xvec::simd::iota<TypeParam>;
    constexpr auto computed = simd_bit_cast<uint8_t>(original);
    BOOST_TEST(computed.size() == simdByteSize);

    // The internal representations should be identical.
    std::array<uint8_t, simdByteSize> originalData;
    std::copy_n((const uint8_t *)&original, simdByteSize, originalData.begin());
    std::array<uint8_t, simdByteSize> computedData;
    partial_store(computed, computedData);
    BOOST_TEST(computedData == originalData);
  }
}
