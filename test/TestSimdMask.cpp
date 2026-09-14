//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimdTestUtilities.hpp"
#include "TypesToTest.hpp"

// Assumes the generator works.
template<typename MASK> constexpr MASK k_emptyMask = MASK(false);
template<typename MASK> constexpr MASK k_fullMask = MASK(true);
template<typename MASK> constexpr MASK k_lsbMask = MASK([](size_t idx){return idx==0; });
template<typename MASK> constexpr MASK k_msbMask = MASK([](size_t idx){return idx==(MASK::size - 1); });

/// Utility function to make it easy to compare two builtin masks. The masks could be narrow
/// (integer) or wide (vector), depending upon the ISA.  The size must be supplied because the
/// builtin rounds up storage to the next allocatable unit, so values which are padding should be
/// ignored.
template<size_t _N, typename BUILTIN_TYPE>
static void ExpectMaskEq(BUILTIN_TYPE lhs, BUILTIN_TYPE rhs)
{
  if constexpr (xvec::simd::has_compact_mask<xvec::simd::target_tag>)
  {
    bool identicalMasks = (lhs == rhs);
    BOOST_TEST(identicalMasks); // boost can't print BitInt values if the test fails, so move the == outside the test.
  }
  else
  {
    using ELEMENT_TYPE = typename std::remove_reference<decltype(BUILTIN_TYPE()[0])>::type;
    constexpr int sizeOfBuiltin = sizeof(BUILTIN_TYPE) / sizeof(ELEMENT_TYPE);

    // Note that only the actual number of mask bits are copied over. The other values can be
    // anything, but are included because builtins are fixed sizes.
    std::array<ELEMENT_TYPE, sizeOfBuiltin> lhsArray = {};
    std::copy_n((const ELEMENT_TYPE*)&lhs, _N, lhsArray.begin());
    std::array<ELEMENT_TYPE, sizeOfBuiltin> rhsArray = {};
    std::copy_n((const ELEMENT_TYPE*)&rhs, _N, rhsArray.begin());

    // Nasty special case - the mask value is all 1's. If this type is really a float (e.g., __m256
    // has a compare mask which is also __m256), then all ones is a NaN. Unfortunately, a NaN never
    // compares true to another NaN. So for float we need to convert NaN to a proper number.
    if constexpr (std::is_floating_point_v<ELEMENT_TYPE>)
    {
      for (auto& e : lhsArray) e = std::isnan(e) ? -1 : 0;
      for (auto& e : rhsArray) e = std::isnan(e) ? -1 : 0;
    }

    BOOST_TEST(lhsArray == rhsArray);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(IsSimdMask, TypeParam, AllSimdMaskTypes)
{
  // Or std::simd, or x86::simd.
  BOOST_TEST(xvec::simd::detail::is_mask_v<TypeParam>);
  //static_assert(simd::is_abi_tag_v<typename TypeParam::abi_type>); // :TODO: Implement

  // Various other types aren't simd masks.
  BOOST_TEST(!xvec::simd::detail::is_mask_v<std::string>);
  BOOST_TEST(!xvec::simd::detail::is_mask_v<double>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseEmpty, TypeParam, AllSimdMaskTypes)
{
  constexpr TypeParam empty = {};
  constexpr typename TypeParam::builtin_type emptyBuiltin = {};
  ExpectMaskEq<TypeParam::size()>(empty.to_builtin(), emptyBuiltin);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(EmptyGetBitset, TypeParam, AllSimdMaskTypes)
{
  constexpr TypeParam empty = {};
  BOOST_TEST(empty.to_bitset().none());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(EmptyGetUllong, TypeParam, AllSimdMaskTypes)
{
  constexpr TypeParam empty = {};
  if constexpr (TypeParam::size() <= 64)
    BOOST_TEST(empty.to_ullong() == 0ULL);
}

// :TODO: Build from boolean value to match std::simd?

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromBuiltin, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  ExpectMaskEq<TypeParam::size()>(fixture.mask0.to_builtin(), fixture.builtin0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FromSameSimdType, TypeParam, AllSimdMaskTypes)
{
  const auto gen = [](size_t idx) -> bool { return (idx % 3) != 0; };
  constexpr auto original = TypeParam(gen);

  // Converting copies don't work yet because llvm has issues with constexpr, but this makes sure that at least
  // copies into the same type will work, which is useful for constexpr versions of other functions.
  constexpr auto computedConstexpr = TypeParam(original);
  ExpectMaskEq<TypeParam::size()>(computedConstexpr.to_builtin(), original.to_builtin());

  auto computedDynamic = TypeParam(std::as_const(original));
  ExpectMaskEq<TypeParam::size()>(computedDynamic.to_builtin(), original.to_builtin());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FromOtherSimdType, TypeParam, AllSimdMaskTypes)
{
  const auto gen = [](size_t idx) -> bool { return (idx % 3) != 0; };

  // Everything can be copied from a signed char mask.
  constexpr auto original = xvec::simd::mask<int8_t, TypeParam::size()>(gen);

  std::bitset<TypeParam::size()> expected;
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = gen(i);

  // constexpr auto computedConstexpr = TypeParam(original);
  // ExpectMaskEq<TypeParam::size()>(computedConstexpr.to_builtin(), original.to_builtin());

  auto computedDynamic = TypeParam(std::as_const(original));
  BOOST_TEST(computedDynamic.to_bitset() == expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructMaskFromBool, TypeParam, AllSimdMaskTypes)
{
  BOOST_TEST(TypeParam(false).to_bitset().none());
  BOOST_TEST(TypeParam(true).to_bitset().all());

  constexpr auto maskFalse = TypeParam(false);
  constexpr auto maskTrue = TypeParam(true);
  BOOST_TEST(maskFalse.to_bitset().none());
  BOOST_TEST(maskTrue.to_bitset().all());

}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructSimdFromMask, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  typename TypeParam::traits::signed_vec_for_mask expected([=](auto i) -> typename TypeParam::traits::signed_vec_for_mask::value_type {
    return fixture.bitset0[i];
  });

  // Construct directly into a simd of the appropriate type.
  {
    typename TypeParam::traits::signed_vec_for_mask computed = fixture.mask0;
    BOOST_SIMD_EQUAL(computed, expected);
  }

  {
    const auto computed = +fixture.mask0;
    BOOST_SIMD_EQUAL(computed, expected);
  }

  {
    const auto computed = -fixture.mask0;
    BOOST_SIMD_EQUAL(computed, -expected);
  }

  {
    // CTAD version, equivalent to +fixture.mask0.
    xvec::simd::basic_vec computed = fixture.mask0;
    BOOST_SIMD_EQUAL(computed, expected);
  }

}

#if defined (_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprConstructSimdFromMask, TypeParam, AllSimdMaskTypes)
{
  constexpr TypeParam mask([](auto i) { return i % 2 == 0; });

  typename TypeParam::traits::signed_vec_for_mask expected([=](auto i) -> typename TypeParam::traits::signed_vec_for_mask::value_type {
    return (i % 2) == 0;
  });

  {
    constexpr typename TypeParam::traits::signed_vec_for_mask computed = mask;
    BOOST_SIMD_EQUAL(computed, expected);
  }

  {
    constexpr auto computed = +mask;
    BOOST_SIMD_EQUAL(computed, expected);
  }

  {
    constexpr auto computed = -mask;
    BOOST_SIMD_EQUAL(computed, -expected);
  }
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(ReadAccessByIndex, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  const auto constMask = TypeParam(fixture.mask0);
  auto nonConstMask = TypeParam(fixture.mask0);

  // Don't remove the loop - it is testing the [] for each index.
  for (std::size_t i=0; i<TypeParam::size(); ++i)
  {
    BOOST_CHECK(constMask[i] == fixture.bitset0[i]);
    BOOST_CHECK(nonConstMask[i] == fixture.bitset0[i]);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprReadAccessByIndex, TypeParam, AllSimdMaskTypes)
{
  constexpr auto v = TypeParam(0b0101010ul);
  BOOST_CHECK(v[0] == false);
  if (TypeParam::size > 4)
    BOOST_CHECK(v[3] == true);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BuiltinGetBitset, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  BOOST_TEST(fixture.mask0.to_bitset() == fixture.bitset0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BuiltinGetUllong, TypeParam, AllSimdMaskTypes)
{
  if constexpr (TypeParam::size() <= 64)
  {
    SimdMaskFixture<TypeParam> fixture;
    BOOST_TEST(fixture.mask0.to_ullong() == fixture.bitset0.to_ullong());

    #if defined(_XVEC_HAS_CONSTEXPR)
      BOOST_TEST(at_compile_time(fixture.every3.to_ullong()) == SimdMaskFixture<TypeParam>::every3.to_ullong());
    #endif
  }
}

BOOST_AUTO_TEST_CASE(WideMaskGetUllong)
{
  constexpr std::size_t N = std::numeric_limits<unsigned long long>::digits;
  using WideMask = xvec::simd::mask<std::uint8_t, 2 * N>;

  // A mask wider than unsigned long long is valid when every high bit is
  // false. Exercise zero, an interior low bit, and the highest bit that can
  // be represented.
  constexpr WideMask empty(false);
  constexpr WideMask lowBits([](std::size_t i) {
    return i == 0 || i == N / 2 || i == N - 1;
  });

  constexpr unsigned long long expected =
    1ULL |
    (1ULL << (N / 2)) |
    (1ULL << (N - 1));

  static_assert(WideMask::size() > N);

  // Ordinary calls exercise the runtime implementation.
  BOOST_TEST(empty.to_ullong() == 0ULL);
  BOOST_TEST(lowBits.to_ullong() == expected);

#if defined(_XVEC_HAS_CONSTEXPR)
  // These calls exercise the constant-evaluation implementation and ensure
  // it never shifts by N or greater.
  static_assert(empty.to_ullong() == 0ULL);
  static_assert(lowBits.to_ullong() == expected);
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BitsetOperatorConversion, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  BOOST_TEST(fixture.mask0.to_bitset() == fixture.bitset0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromBitset, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  auto mask = TypeParam(fixture.bitset0);
  ExpectMaskEq<TypeParam::size()>(mask.to_builtin(), fixture.builtin0);

  // Constexpr bitset.
  if constexpr (TypeParam::size < 64)
  {
    constexpr uint64_t cRawBits = 0b1010101010101u & ((uint64_t(1) << TypeParam::size) - 1);
    constexpr std::bitset<TypeParam::size> cBitset(cRawBits);
    constexpr auto cMask = TypeParam(cBitset);
    BOOST_TEST(cMask.to_ullong() == cRawBits);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromUnsigned, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  // Remove all but the lowest 64-bits since that is all that can be set or tested.
  auto truncatedBitset = fixture.bitset0;
  for (std::size_t i=64; i<TypeParam::size(); ++i)
    truncatedBitset.reset(i);

  const auto v = truncatedBitset.to_ullong();
  TypeParam mask(v);

  const auto truncatedBuiltin = getMaskFromBitset<typename TypeParam::builtin_type, SimdMaskFixture<TypeParam>::k_numBits>(truncatedBitset);
  ExpectMaskEq<TypeParam::size()>(mask.to_builtin(), truncatedBuiltin);

  // Constexpr unsigned.
  if constexpr (TypeParam::size < 64)
  {
    constexpr uint64_t cRawBits = 0b1010101010101u & ((uint64_t(1) << TypeParam::size) - 1);
    constexpr auto cMask = TypeParam(cRawBits);
    BOOST_TEST(cMask.to_ullong() == cRawBits);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprInitialiseFromUnsigned, TypeParam, AllSimdMaskTypes)
{
  // Create some constexpr bits, where bits outside the valid size are cleared.
  constexpr uint64_t bits = 0xAAAAAAAAAAAAAAAA & ((uint64_t(1) << std::min<int>(TypeParam::size(), 63)) - 1);
  std::bitset<TypeParam::size()> expected(bits);

  constexpr TypeParam computed(bits);
  BOOST_TEST(computed.to_bitset() == expected);

  if constexpr (TypeParam::size() <= 64)
    BOOST_TEST(computed.to_ullong() == bits);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromGenerator, TypeParam, AllSimdMaskTypes)
{
  // Generate values to fill the mask using the following function, which converts an index into a value.
  constexpr auto gen = [](size_t idx) -> bool { return (idx % 3) == 0;};

  auto expected = std::bitset<TypeParam::size()>();
  for (size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = gen(i);

  const auto computedDynamic = TypeParam(gen);
  constexpr auto computedConstexpr = TypeParam(gen);

  BOOST_TEST(computedDynamic.to_bitset() == expected);
  BOOST_TEST(computedConstexpr.to_bitset() == expected);

  if constexpr (TypeParam::size() <= 64)
  {
    BOOST_TEST(computedDynamic.to_ullong() == expected.to_ullong());
    BOOST_TEST(computedConstexpr.to_ullong() == expected.to_ullong());
  }

  // Check the generator works with a type which is narrower than the mask too.
  constexpr uint32_t narrowBits = 0x80000001;
  constexpr TypeParam narrowComputed(narrowBits);
  BOOST_TEST(narrowComputed.to_bitset() == std::bitset<TypeParam::size()>(narrowBits));

#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr TypeParam narrowComputed_ce(narrowBits);
  BOOST_TEST(narrowComputed_ce.to_bitset() == std::bitset<TypeParam::size()>(narrowBits));
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseNbitMask, TypeParam, AllSimdMaskTypes)
{
  using xvec::simd::mask_from_count;
  using VecType = typename TypeParam::traits::signed_vec_for_mask;

  // Generate a mask represent the bottom N elements of the mask, where N is 1/3rd for this test.
  constexpr int N = TypeParam::size() / 3;

  auto expected = std::bitset<TypeParam::size()>();
  for (size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = (i < N);

  const auto computedDynamic = mask_from_count<VecType>(N);
  constexpr auto computedConstexpr = mask_from_count<VecType>(N);

  BOOST_TEST(computedDynamic.to_bitset() == expected);
  BOOST_TEST(computedConstexpr.to_bitset() == expected);

  // Special cases.
  BOOST_TEST(mask_from_count<VecType>(0).to_bitset() == TypeParam().to_bitset());                      // Zero should give empty mask.
  BOOST_TEST(mask_from_count<VecType>(TypeParam::size).to_bitset() == (TypeParam(true)).to_bitset());     // Mask size should give full mask.
  BOOST_TEST(mask_from_count<VecType>(TypeParam::size + 3).to_bitset() == (TypeParam(true)).to_bitset()); // Bigger should give full mask.
  BOOST_TEST(mask_from_count<VecType>(1024).to_bitset() == (TypeParam(true)).to_bitset());                // Bigger should give full mask.
  BOOST_TEST(mask_from_count<VecType>(845856).to_bitset() == (TypeParam(true)).to_bitset());              // Bigger should give full mask.

  // Very special case. 128 mask in a single byte element can cause issues.
  using AwkwardMaskType = xvec::simd::mask<uint8_t, 128>;
  using AwkwardType = xvec::simd::vec<uint8_t, 128>;
  BOOST_TEST(mask_from_count<AwkwardType>(845856).to_bitset() == (AwkwardMaskType(true)).to_bitset());
}

// Essentially identical to the n_elements test above, but for mask_from_count.
// Once n_elements is removed the test above can be removed too.
BOOST_AUTO_TEST_CASE_TEMPLATE(MaskFromCount, TypeParam, AllSimdTypes)
{
  using xvec::simd::mask_from_count;
  using _Mp = typename TypeParam::mask_type;

  // Generate a mask represent the bottom N elements of the mask, where N is 1/3rd for this test.
  constexpr int N = TypeParam::size() / 3;

  auto expected = std::bitset<TypeParam::size()>();
  for (size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = (i < N);

  const auto computedDynamic = mask_from_count<TypeParam>(N);
  constexpr auto computedConstexpr = mask_from_count<TypeParam>(N);

  BOOST_TEST(computedDynamic.to_bitset() == expected);
  BOOST_TEST(computedConstexpr.to_bitset() == expected);

  // Special cases.
  BOOST_TEST(mask_from_count<TypeParam>(0).to_bitset() == _Mp().to_bitset());                      // Zero should give empty mask.
  BOOST_TEST(mask_from_count<TypeParam>(TypeParam::size).to_bitset() == (_Mp(true)).to_bitset());     // Mask size should give full mask.
  BOOST_TEST(mask_from_count<TypeParam>(TypeParam::size + 3).to_bitset() == (_Mp(true)).to_bitset()); // Bigger should give full mask.
  BOOST_TEST(mask_from_count<TypeParam>(1024).to_bitset() == (_Mp(true)).to_bitset());                // Bigger should give full mask.
  BOOST_TEST(mask_from_count<TypeParam>(845856).to_bitset() == (_Mp(true)).to_bitset());              // Bigger should give full mask.

  // Very special case. 128 mask in a single byte element can cause issues.
  using BigType = xvec::simd::vec<char, 128>;
  BOOST_TEST(mask_from_count<BigType>(845856).to_bitset() == (BigType() == BigType()).to_bitset());

  // Scalar (simd-generic) version.
  BOOST_TEST(mask_from_count<float>(0) == false);
  BOOST_TEST(mask_from_count<float>(1) == true);
  BOOST_TEST(mask_from_count<float>(93942) == true);

}

BOOST_AUTO_TEST_CASE_TEMPLATE(StringOutput, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  std::ostringstream computed;
  computed << fixture.mask0;

  std::ostringstream expected;
  expected << fixture.bitset0;

  BOOST_TEST(computed.str() == expected.str());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskCompare, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  // Create a mask which is similar, but has toggled bits.
  const auto toggle = TypeParam([](auto idx) { return idx % 5 == 0; });
  auto similar = fixture.mask0 ^ toggle;

  BOOST_TEST((fixture.mask0 == fixture.mask0).to_bitset().all());
  BOOST_TEST((fixture.mask0 == (!fixture.mask0)).to_bitset().none());
  BOOST_TEST((fixture.mask0 == similar).to_bitset() == ~(toggle.to_bitset()));

  BOOST_TEST((fixture.mask0 != (!fixture.mask0)).to_bitset().all());
  BOOST_TEST((fixture.mask0 != (fixture.mask0)).to_bitset().none());
  BOOST_TEST((fixture.mask0 != similar).to_bitset() == (toggle).to_bitset());

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(at_compile_time(fixture.every3 == fixture.every3).to_bitset().all());
  BOOST_TEST(at_compile_time(fixture.every3 != fixture.every3).to_bitset().none());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskRelationalCompare, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  const auto toggle = TypeParam([](auto idx) { return idx % 5 == 0; });
  const auto similar = fixture.mask0 ^ toggle;
  const auto expected_lt = applyBinaryToArray(fixture.mask0, similar, std::less{});
  const auto expected_le = applyBinaryToArray(fixture.mask0, similar, std::less_equal{});
  const auto expected_gt = applyBinaryToArray(fixture.mask0, similar, std::greater{});
  const auto expected_ge = applyBinaryToArray(fixture.mask0, similar, std::greater_equal{});

  BOOST_TEST((fixture.mask0 < fixture.mask0).to_bitset().none());
  BOOST_TEST((fixture.mask0 <= fixture.mask0).to_bitset().all());
  BOOST_TEST((fixture.mask0 > fixture.mask0).to_bitset().none());
  BOOST_TEST((fixture.mask0 >= fixture.mask0).to_bitset().all());

  BOOST_TEST(to_array(fixture.mask0 < similar) == expected_lt, boost::test_tools::per_element());
  BOOST_TEST(to_array(fixture.mask0 <= similar) == expected_le, boost::test_tools::per_element());
  BOOST_TEST(to_array(fixture.mask0 > similar) == expected_gt, boost::test_tools::per_element());
  BOOST_TEST(to_array(fixture.mask0 >= similar) == expected_ge, boost::test_tools::per_element());
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprMaskRelationalCompare, TypeParam, AllSimdMaskTypes)
{
  constexpr auto lhs = k_lsbMask<TypeParam>;
  constexpr auto rhs = k_msbMask<TypeParam>;
  constexpr auto expected_lt = applyBinaryToArray(lhs, rhs, std::less{});
  constexpr auto expected_le = applyBinaryToArray(lhs, rhs, std::less_equal{});
  constexpr auto expected_gt = applyBinaryToArray(lhs, rhs, std::greater{});
  constexpr auto expected_ge = applyBinaryToArray(lhs, rhs, std::greater_equal{});

  BOOST_TEST(to_array(at_compile_time(lhs < lhs)) == applyBinaryToArray(lhs, lhs, std::less{}),
             boost::test_tools::per_element());
  BOOST_TEST(to_array(at_compile_time(lhs <= lhs)) == applyBinaryToArray(lhs, lhs, std::less_equal{}),
             boost::test_tools::per_element());
  BOOST_TEST(to_array(at_compile_time(lhs > lhs)) == applyBinaryToArray(lhs, lhs, std::greater{}),
             boost::test_tools::per_element());
  BOOST_TEST(to_array(at_compile_time(lhs >= lhs)) == applyBinaryToArray(lhs, lhs, std::greater_equal{}),
             boost::test_tools::per_element());

  BOOST_TEST(to_array(at_compile_time(lhs < rhs)) == expected_lt, boost::test_tools::per_element());
  BOOST_TEST(to_array(at_compile_time(lhs <= rhs)) == expected_le, boost::test_tools::per_element());
  BOOST_TEST(to_array(at_compile_time(lhs > rhs)) == expected_gt, boost::test_tools::per_element());
  BOOST_TEST(to_array(at_compile_time(lhs >= rhs)) == expected_ge, boost::test_tools::per_element());
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskLogicalNot, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  const auto computed = !fixture.mask0;
  BOOST_TEST(computed.to_bitset() == ~fixture.bitset0);

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(at_compile_time(!fixture.every3).to_bitset() == ~fixture.every3.to_bitset());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskBitwiseNot, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  auto expected = applyUnaryToArray(fixture.mask0, std::bit_not{});

  BOOST_TEST(to_array(~fixture.mask0) == expected, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(to_array(at_compile_time(~fixture.every3)) ==
             applyUnaryToArray(fixture.every3, std::bit_not{}),
             boost::test_tools::per_element());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskLogicalAnd, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  const auto computed = fixture.mask0 && fixture.mask1;
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 & fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(at_compile_time(fixture.every3 && !fixture.every3).to_bitset().none());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskLogicalOr, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  const auto computed = fixture.mask0 || fixture.mask1;
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 | fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(at_compile_time(fixture.every3 || !fixture.every3).to_bitset().all());
#endif
}


BOOST_AUTO_TEST_CASE_TEMPLATE(MaskAnd, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  const auto computed = fixture.mask0 & fixture.mask1;
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 & fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(at_compile_time(fixture.every3 & !fixture.every3).to_bitset().none());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskOr, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  const auto computed = fixture.mask0 | fixture.mask1;
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 | fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(at_compile_time(fixture.every3 | !fixture.every3).to_bitset().all());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskXor, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  const auto computed = fixture.mask0 ^ fixture.mask1;
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 ^ fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  BOOST_TEST(at_compile_time(fixture.every3 ^ fixture.every3).to_bitset().none());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskAssignAnd, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  auto computed = fixture.mask0;
  const auto assignOut = (computed &= fixture.mask1);
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 & fixture.bitset1));
  BOOST_TEST(assignOut.to_bitset() == (fixture.bitset0 & fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr auto constexprComputed = [] {
    auto mask = SimdMaskFixture<TypeParam>::every3;
    return mask &= !mask;
  }();
  BOOST_TEST(at_compile_time(constexprComputed).to_bitset().none());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskAssignOr, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  auto computed = fixture.mask0;
  const auto assignOut = (computed |= fixture.mask1);
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 | fixture.bitset1));
  BOOST_TEST(assignOut.to_bitset() == (fixture.bitset0 | fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr auto constexprComputed = [] {
    auto mask = SimdMaskFixture<TypeParam>::every3;
    return mask |= !mask;
  }();
  BOOST_TEST(at_compile_time(constexprComputed).to_bitset().all());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskAssignXor, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;
  auto computed = fixture.mask0;
  const auto assignOut = (computed ^= fixture.mask1);
  BOOST_TEST(computed.to_bitset() == (fixture.bitset0 ^ fixture.bitset1));
  BOOST_TEST(assignOut.to_bitset() == (fixture.bitset0 ^ fixture.bitset1));

#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr auto constexprComputed = [] {
    auto mask = SimdMaskFixture<TypeParam>::every3;
    return mask ^= mask;
  }();
  BOOST_TEST(at_compile_time(constexprComputed).to_bitset().none());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AllOfBool, TypeParam, AllSimdMaskTypes)
{
  BOOST_TEST(xvec::simd::all_of(false) == false);
  BOOST_TEST(xvec::simd::all_of(true) == true);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AllOfMask, TypeParam, AllSimdMaskTypes)
{
  // non-constexpr
  BOOST_TEST(!all_of(k_emptyMask<TypeParam>));

  if constexpr (TypeParam::size > 1)
  {
    BOOST_TEST(!all_of(k_lsbMask<TypeParam>));
    BOOST_TEST(!all_of(k_msbMask<TypeParam>));
    BOOST_TEST( all_of(k_fullMask<TypeParam>));

    // constexpr
    # if defined(_XVEC_HAS_CONSTEXPR)
      BOOST_TEST(!at_compile_time(all_of(k_emptyMask<TypeParam>)));
      BOOST_TEST(!at_compile_time(all_of(k_lsbMask<TypeParam>)));
      BOOST_TEST(!at_compile_time(all_of(k_msbMask<TypeParam>)));
      BOOST_TEST( at_compile_time(all_of(k_fullMask<TypeParam>)));
    #endif
  }
  else
  {
    // Scalar
    BOOST_TEST(all_of(k_lsbMask<TypeParam>));
    BOOST_TEST(all_of(k_msbMask<TypeParam>));
    BOOST_TEST(all_of(k_fullMask<TypeParam>));

    // constexpr
    # if defined(_XVEC_HAS_CONSTEXPR)
      BOOST_TEST(at_compile_time(all_of(k_lsbMask<TypeParam>)));
      BOOST_TEST(at_compile_time(all_of(k_msbMask<TypeParam>)));
      BOOST_TEST(at_compile_time(all_of(k_fullMask<TypeParam>)));
    #endif
  }

}

BOOST_AUTO_TEST_CASE_TEMPLATE(AnyOfBool, TypeParam, AllSimdMaskTypes)
{
  BOOST_TEST(xvec::simd::any_of(false) == false);
  BOOST_TEST(xvec::simd::any_of(true) == true);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AnyOfSimd, TypeParam, AllSimdMaskTypes)
{
  // Non-constexpr
  BOOST_TEST(!any_of(k_emptyMask<TypeParam>));
  BOOST_TEST( any_of(k_lsbMask<TypeParam>));
  BOOST_TEST( any_of(k_msbMask<TypeParam>));
  BOOST_TEST( any_of(k_fullMask<TypeParam>));

  // Constexpr
  # if defined(_XVEC_HAS_CONSTEXPR)
    BOOST_TEST(!at_compile_time(any_of(k_emptyMask<TypeParam>)));
    BOOST_TEST( at_compile_time(any_of(k_lsbMask<TypeParam>)));
    BOOST_TEST( at_compile_time(any_of(k_msbMask<TypeParam>)));
    BOOST_TEST( at_compile_time(any_of(k_fullMask<TypeParam>)));
  #endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(NoneOfMask, TypeParam, AllSimdMaskTypes)
{
  BOOST_TEST(xvec::simd::none_of(false) == true);
  BOOST_TEST(xvec::simd::none_of(true) == false);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(NoneOfSimd, TypeParam, AllSimdMaskTypes)
{
  // Non-constexpr
  BOOST_TEST( none_of(k_emptyMask<TypeParam>));
  BOOST_TEST(!none_of(k_lsbMask<TypeParam>));
  BOOST_TEST(!none_of(k_msbMask<TypeParam>));
  BOOST_TEST(!none_of(k_fullMask<TypeParam>));

  // Constexpr
  # if defined(_XVEC_HAS_CONSTEXPR)
    BOOST_TEST( at_compile_time(none_of(k_emptyMask<TypeParam>)));
    BOOST_TEST(!at_compile_time(none_of(k_lsbMask<TypeParam>)));
    BOOST_TEST(!at_compile_time(none_of(k_msbMask<TypeParam>)));
    BOOST_TEST(!at_compile_time(none_of(k_fullMask<TypeParam>)));
  #endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceCountBool, TypeParam, AllSimdMaskTypes)
{
  BOOST_TEST(xvec::simd::reduce_count(false) == 0);
  BOOST_TEST(xvec::simd::reduce_count(true) == 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceCountSimd, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  // Non-constexpr predefined constants
  BOOST_TEST(reduce_count(k_emptyMask<TypeParam>) == 0);
  BOOST_TEST(reduce_count(k_lsbMask<TypeParam>) == 1);
  BOOST_TEST(reduce_count(k_msbMask<TypeParam>) == 1);
  BOOST_TEST(reduce_count(k_fullMask<TypeParam>) == TypeParam::size());

  // Random mask, both constexpr and non-constexpr
  BOOST_TEST(reduce_count(fixture.mask0) == fixture.bitset0.count());

  // Constexpr
  # if defined(_XVEC_HAS_CONSTEXPR)
    BOOST_TEST(at_compile_time(reduce_count(k_emptyMask<TypeParam>)) == 0);
    BOOST_TEST(at_compile_time(reduce_count(k_lsbMask<TypeParam>)) == 1);
    BOOST_TEST(at_compile_time(reduce_count(k_msbMask<TypeParam>)) == 1);
    BOOST_TEST(at_compile_time(reduce_count(k_fullMask<TypeParam>)) == TypeParam::size());

    constexpr auto ceMask = GetConstexprRandomVector<TypeParam, 0>();
    BOOST_TEST(at_compile_time(reduce_count(ceMask)) == ceMask.to_bitset().count());
  #endif

}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceMinIndex, TypeParam, AllSimdMaskTypes)
{
  // Non-constexpr predefined constants
  BOOST_TEST(reduce_min_index(k_lsbMask<TypeParam>) == 0);
  if (TypeParam::size > 1) BOOST_TEST(reduce_min_index(!k_lsbMask<TypeParam>) == 1);
  BOOST_TEST(reduce_min_index(k_msbMask<TypeParam>) == (TypeParam::size - 1));
  BOOST_TEST(reduce_min_index(k_lsbMask<TypeParam> | k_msbMask<TypeParam>) == 0);
  BOOST_TEST(reduce_min_index(k_fullMask<TypeParam>) == 0);

  // Random mask. The reduction only works if at least one bit is set. not a
  // problem for the larger sizes, but a random small mask might just happen to
  // be empty.
  SimdMaskFixture<TypeParam> fixture;
  if (any_of(fixture.mask0))
  {
    auto expected = 0;
    while (expected <= (TypeParam::size - 1) && !fixture.bitset0[expected]) expected += 1;
    BOOST_TEST(reduce_min_index(fixture.mask0) == expected);
  }

  // Constexpr predefined constants
  #if defined(_XVEC_HAS_CONSTEXPR)
    BOOST_TEST(at_compile_time(reduce_min_index(k_lsbMask<TypeParam>)) == 0);
    BOOST_TEST(at_compile_time(reduce_min_index(!k_lsbMask<TypeParam>)) == 1);
    BOOST_TEST(at_compile_time(reduce_min_index(k_msbMask<TypeParam>)) == (TypeParam::size - 1));
    BOOST_TEST(at_compile_time(reduce_min_index(k_lsbMask<TypeParam> | k_msbMask<TypeParam>)) == 0);
    BOOST_TEST(at_compile_time(reduce_min_index(k_fullMask<TypeParam>)) == 0);

    constexpr auto ceMask = GetConstexprRandomVector<TypeParam, 0>();
    if constexpr (any_of(ceMask))
    {
      auto ceExpected = 0;
      while (ceExpected <= (TypeParam::size - 1) && !ceMask[ceExpected])
        ceExpected += 1;
      BOOST_TEST(at_compile_time(reduce_min_index(ceMask)) == ceExpected);
    }
  #endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceMaxIndex, TypeParam, AllSimdMaskTypes)
{
  // Non-constexpr predefined constants
  BOOST_TEST(reduce_max_index(k_lsbMask<TypeParam>) == 0);
  BOOST_TEST(reduce_max_index(k_msbMask<TypeParam>) == (TypeParam::size - 1));
  BOOST_TEST(reduce_max_index(k_lsbMask<TypeParam> | k_msbMask<TypeParam>) == (TypeParam::size - 1));
  BOOST_TEST(reduce_max_index(k_fullMask<TypeParam>) == (TypeParam::size - 1));

  // Random mask. The reduction only works if at least one bit is set. not a
  // problem for the larger sizes, but a random small mask might just happen to
  // be empty.
  SimdMaskFixture<TypeParam> fixture;
  if (any_of(fixture.mask0))
  {
    int expected = TypeParam::size - 1;
    while (expected >= 0 && !fixture.bitset0[expected]) expected -= 1;
    BOOST_TEST(reduce_max_index(fixture.mask0) == expected);
  }

  // Constexpr predefined constants
  #if defined(_XVEC_HAS_CONSTEXPR)
    BOOST_TEST(at_compile_time(reduce_max_index(k_lsbMask<TypeParam>)) == 0);
    BOOST_TEST(at_compile_time(reduce_max_index(k_msbMask<TypeParam>)) == (TypeParam::size - 1));
    BOOST_TEST(at_compile_time(reduce_max_index(k_lsbMask<TypeParam> | k_msbMask<TypeParam>)) == (TypeParam::size - 1));
    BOOST_TEST(at_compile_time(reduce_max_index(k_fullMask<TypeParam>)) == (TypeParam::size - 1));

    constexpr auto ceMask = GetConstexprRandomVector<TypeParam, 0>();
    if constexpr (any_of(ceMask))
    {
      int ceExpected = TypeParam::size - 1;
      while (ceExpected >= 0 && !ceMask[ceExpected])
        ceExpected -= 1;
      BOOST_TEST(at_compile_time(reduce_max_index(ceMask)) == ceExpected);
    }
  #endif

}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceIndexBool, TypeParam, AllSimdMaskTypes)
{
  // Scalar bool versions. Note that everything is zero, which is because a 
  // precondition specifies that the argument is always true, so the spec. is
  // allowed to say that the result is never anything other than 0!
  BOOST_TEST(xvec::simd::reduce_min_index(false) == 0);
  BOOST_TEST(xvec::simd::reduce_min_index(true) == 0);
  BOOST_TEST(xvec::simd::reduce_max_index(false) == 0);
  BOOST_TEST(xvec::simd::reduce_max_index(true) == 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskSimdSelect, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> f;
  SimdMaskFixture<TypeParam> mf; // mask fixture only has 2 masks, so create a third here to act as the selector.

  std::bitset<TypeParam::size> expected = {};
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = mf.bitset0[i] ? f.bitset0[i] : f.bitset1[i];

  auto computed = select(mf.mask0, f.mask0, f.mask1);
  BOOST_TEST(computed.to_bitset() == expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskSimdSelectForBool, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> mf;

  auto computed = select(mf.mask0, false, true); // Equivalent to a not.
  BOOST_TEST(computed.to_bitset() == ~((mf.mask0).to_bitset()));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(Insert, TypeParam, AllSimdMaskTypes)
{
  SimdMaskFixture<TypeParam> fixture;

  if constexpr (TypeParam::size() > 4)
  {
    // Insert 2 elements starting at offset 0.
    const auto expected = setBitSubset<0, 2>(fixture.bitset0, fixture.bitset1);
    const auto computed = xvec::simd::insert<0>(fixture.mask0, take<2>(fixture.mask1));
    BOOST_TEST(computed.to_bitset() == expected);
  }

  if constexpr (TypeParam::size() > 4)
  {
    const auto expected = setBitSubset<1, 4>(fixture.bitset0, fixture.bitset1);
    const auto computed = xvec::simd::insert<1>(fixture.mask0, take<3>(fixture.mask1));
    BOOST_TEST(computed.to_bitset() == expected);
  }

  if constexpr (TypeParam::size() >= 8)
  {
    const auto expected = setBitSubset<TypeParam::size() - 5, TypeParam::size()>(fixture.bitset0, fixture.bitset1);
    const auto computed = xvec::simd::insert<TypeParam::size() - 5>(fixture.mask0, take<5>(fixture.mask1));
    BOOST_TEST(computed.to_bitset() == expected);
  }
}
