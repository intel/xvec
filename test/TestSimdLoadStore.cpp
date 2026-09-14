//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimdTestUtilities.hpp"
#include "TypesToTest.hpp"

namespace {

/// Returns a std::array filled with std::iota starting from `Start`.
template <typename T, std::size_t Count>
constexpr auto getMemoryBlock()
{
  std::array<T, Count> block{};
  std::iota(block.begin(), block.end(), T());
  return block;
}

/// Returns a std::array with the first `N` elements filled via iota from `Start`,
/// and the remainder value-initialised.
template <typename Vec, std::size_t N = Vec::size, int Start = 3>
constexpr auto getExpectedBlock()
{
  std::array<typename Vec::value_type, Vec::size> expected = {};
  for (std::size_t i = 0; i < N; ++i)
    expected[i] = static_cast<typename Vec::value_type>(Start + i);
  return expected;
}

/// Returns a std::array with the first `N` elements filled via iota from `Start`,
/// but with odd-indexed elements value-initialised (i.e., only even indices retain values).
template <typename Vec, std::size_t N = Vec::size, int Start = 3>
constexpr auto getExpectedEvenBlock()
{
  std::array<typename Vec::value_type, Vec::size> expected = {};
  for (std::size_t i = 0; i < N; ++i)
    if (i % 2 == 0)
      expected[i] = static_cast<typename Vec::value_type>(Start + i);
  return expected;
}

/// Return a mask for the given vector which contains all the even bits only.
template<typename Vec>
constexpr auto getEvenMask() {
  return typename Vec::mask_type([](size_t idx) -> bool { return (idx % 2) == 0;});
}

template <typename T, std::size_t N>
constexpr auto with_iota_window(std::array<T, N> base, std::size_t offset, std::size_t count)
{
  for (std::size_t i = 0; i < count; ++i)
    base[offset + i] = static_cast<T>(i);
  return base;
}

template <typename T, std::size_t N>
constexpr auto with_iota_window_even(std::array<T, N> base, std::size_t offset, std::size_t count)
{
  if (offset > N)
    return base;

  const std::size_t limit = N - offset;
  if (count > limit)
    count = limit;

  for (std::size_t i = 0; i < count; ++i)
    if ((i % 2) == 0)
      base[offset + i] = static_cast<T>(i);

  return base;
}

} // anonymous namespace

// :TODO: The tests in this file assume that the values are loaded from real-valued, so
// what should happen when a complex value is loaded from a real-valued source.
// Should it convert, or break? For the moment it breaks.
using LoadableSimdTypes = ArithmeticSimdTypes;

/// Check that the default template parameter can be deduced properly.
using LoadableNativeTypes = boost::mp11::mp_transform<xvec::simd::basic_vec, UnsignedIntegerTypes>;

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Constructor_Inside_Unmasked_NoConvert_DeducedVec, TypeParam, LoadableSimdTypes)
{
  static constexpr auto cvalues = getMemoryBlock<unsigned short, TypeParam::size + 64>();
  static constexpr auto cexpected = getExpectedBlock<xvec::simd::rebind_t<unsigned short, TypeParam>>();

#if defined(_XVEC_HAS_CONSTEXPR)
  static_assert([] {
    constexpr xvec::simd::basic_vec computed =
      std::span(cvalues).subspan(3).template first<TypeParam::size>();
    return to_array(computed) == cexpected;
  }());
#endif

  auto values = cvalues;
  auto expected = cexpected;
  xvec::simd::basic_vec computed = std::span(values).subspan(3).template first<TypeParam::size>();
  BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Unchecked_Inside_Unmasked_Convert_ExplicitVec, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::unchecked_load;

  static constexpr auto cvalues = getMemoryBlock<unsigned short, TypeParam::size + 64>();
  static constexpr auto cexpected = getExpectedBlock<TypeParam>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(3);

  auto values = cvalues;
  auto expected = cexpected;

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)    
    static_assert([] {
      constexpr auto v = unchecked_load<TypeParam>(cvalues.begin() + 3, cvalues.begin() + 3 + TypeParam::size, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = unchecked_load<TypeParam>(values.begin() + 3, values.begin() + 3 + TypeParam::size, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = unchecked_load<TypeParam>(cvalues.begin() + 3, TypeParam::size, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = unchecked_load<TypeParam>(values.begin() + 3, TypeParam::size, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }

  // Range
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = unchecked_load<TypeParam>(crange, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(3);
    auto computedFromRange = unchecked_load<TypeParam>(r, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Unchecked_Inside_Unmasked_Convert_DeducedVec, TypeParam, LoadableNativeTypes)
{
  using xvec::simd::unchecked_load;

  static constexpr auto cvalues = getMemoryBlock<typename TypeParam::value_type, TypeParam::size + 64>();
  static constexpr auto cexpected = getExpectedBlock<TypeParam>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(3);

  auto values = cvalues;
  auto expected = cexpected;

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = unchecked_load(cvalues.begin() + 3, cvalues.begin() + 3 + TypeParam::size, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = unchecked_load(values.begin() + 3, values.begin() + 3 + TypeParam::size, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = unchecked_load(cvalues.begin() + 3, TypeParam::size, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = unchecked_load(values.begin() + 3, TypeParam::size, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }

  // Range
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = unchecked_load(crange, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(3);
    auto computedFromRange = unchecked_load(r, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Unchecked_Inside_Masked_Convert_ExplicitVec, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::unchecked_load;

  static constexpr auto cvalues = getMemoryBlock<unsigned short, TypeParam::size + 64>();
  static constexpr auto cexpected = getExpectedEvenBlock<TypeParam>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(3);
  constexpr auto evenMask = getEvenMask<TypeParam>();

  auto values = cvalues;
  auto expected = cexpected;

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load<TypeParam>(cvalues.begin() + 3, cvalues.begin() + 3 + TypeParam::size, evenMask, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = unchecked_load<TypeParam>(values.begin() + 3, values.begin() + 3 + TypeParam::size, evenMask, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load<TypeParam>(cvalues.begin() + 3, TypeParam::size, evenMask, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = unchecked_load<TypeParam>(values.begin() + 3, TypeParam::size, evenMask, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }

  // Range
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load<TypeParam>(crange, evenMask, xvec::simd::flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(3);
    auto computedFromRange = unchecked_load<TypeParam>(r, evenMask, xvec::simd::flag_convert);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Unchecked_Inside_Masked_NoConvert_ExplicitVec, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::unchecked_load;

  static constexpr auto cvalues = getMemoryBlock<typename TypeParam::value_type, TypeParam::size + 64>();
  static constexpr auto cexpected = getExpectedEvenBlock<TypeParam>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(3);
  constexpr auto evenMask = getEvenMask<TypeParam>();

  auto values = cvalues;
  auto expected = cexpected;

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load<TypeParam>(cvalues.begin() + 3, cvalues.begin() + 3 + TypeParam::size, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = unchecked_load<TypeParam>(values.begin() + 3, values.begin() + 3 + TypeParam::size, evenMask);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load<TypeParam>(cvalues.begin() + 3, TypeParam::size, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = unchecked_load<TypeParam>(values.begin() + 3, TypeParam::size, evenMask);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }

  // Range
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load<TypeParam>(crange, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(3);
    auto computedFromRange = unchecked_load<TypeParam>(r, evenMask);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Unchecked_Inside_Masked_NoConvert_DeducedVec, TypeParam, LoadableNativeTypes)
{
  using xvec::simd::unchecked_load;

  static constexpr auto cvalues = getMemoryBlock<typename TypeParam::value_type, TypeParam::size + 64>();
  static constexpr auto cexpected = getExpectedEvenBlock<TypeParam>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(3);
  constexpr auto evenMask = getEvenMask<TypeParam>();

  auto values = cvalues;
  auto expected = cexpected;

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load(cvalues.begin() + 3, cvalues.begin() + 3 + TypeParam::size, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = unchecked_load(values.begin() + 3, values.begin() + 3 + TypeParam::size, evenMask);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load(cvalues.begin() + 3, TypeParam::size, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = unchecked_load(values.begin() + 3, TypeParam::size, evenMask);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }

  // Range
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = unchecked_load(crange, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(3);
    auto computedFromRange = unchecked_load(r, evenMask);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Partial_Outside_Unmasked_Convert_ExplicitVec, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::partial_load;
  using xvec::simd::flag_convert;

  constexpr auto srcSize = TypeParam::size - 1;
  constexpr auto offset = (TypeParam::size + 64) - srcSize;
  static constexpr auto cvalues = getMemoryBlock<unsigned short, TypeParam::size + 64>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(offset);

  static constexpr auto cexpected = [] {
    std::array<typename TypeParam::value_type, TypeParam::size> out{};
    for (std::size_t i = 0; i < srcSize; ++i)
      out[i] = static_cast<typename TypeParam::value_type>(cvalues[offset + i]);
    return out;
  }();

  auto values = cvalues;
  auto expected = cexpected;

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load<TypeParam>(cvalues.begin() + offset, cvalues.begin() + offset + srcSize, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = partial_load<TypeParam>(values.begin() + offset, values.begin() + offset + srcSize, flag_convert);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load<TypeParam>(cvalues.begin() + offset, srcSize, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = partial_load<TypeParam>(values.begin() + offset, srcSize, flag_convert);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }

  // Range
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load<TypeParam>(crange, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(offset);
    auto computedFromRange = partial_load<TypeParam>(r, flag_convert);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Partial_Outside_Unmasked_Convert_DeducedVec, TypeParam, LoadableNativeTypes)
{
  using xvec::simd::partial_load;
  using xvec::simd::flag_convert;
  using value_type = typename TypeParam::value_type;

  constexpr auto srcSize = TypeParam::size - 1;
  constexpr auto offset = (TypeParam::size + 64) - srcSize;
  static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(offset);

  static constexpr auto cexpected = [] {
    std::array<value_type, TypeParam::size> out{};
    for (std::size_t i = 0; i < srcSize; ++i)
      out[i] = static_cast<value_type>(cvalues[offset + i]);
    return out;
  }();

  auto values = cvalues;
  auto expected = cexpected;

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load(cvalues.begin() + offset, cvalues.begin() + offset + srcSize, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = partial_load(values.begin() + offset, values.begin() + offset + srcSize, flag_convert);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load(cvalues.begin() + offset, srcSize, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = partial_load(values.begin() + offset, srcSize, flag_convert);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }

  // Range
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load(crange, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(offset);
    auto computedFromRange = partial_load(r, flag_convert);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Partial_Outside_Masked_Convert_ExplicitVec, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::partial_load;
  using xvec::simd::flag_convert;

  constexpr auto srcSize = TypeParam::size - 1;
  constexpr auto offset = (TypeParam::size + 64) - srcSize;
  constexpr auto evenMask = getEvenMask<TypeParam>();
  static constexpr auto cvalues = getMemoryBlock<unsigned short, TypeParam::size + 64>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(offset);

  static constexpr auto cexpected = [] {
    std::array<typename TypeParam::value_type, TypeParam::size> out{};
    for (std::size_t i = 0; i < srcSize; ++i)
      if ((i % 2) == 0)
        out[i] = static_cast<typename TypeParam::value_type>(cvalues[offset + i]);
    return out;
  }();

  auto values = cvalues;
  auto expected = cexpected;

  // Masked load from range.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load<TypeParam>(crange, evenMask, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(offset);
    auto computedFromRange = partial_load<TypeParam>(r, evenMask, flag_convert);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load<TypeParam>(cvalues.begin() + offset, cvalues.begin() + offset + srcSize, evenMask, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = partial_load<TypeParam>(values.begin() + offset, values.begin() + offset + srcSize, evenMask, flag_convert);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load<TypeParam>(cvalues.begin() + offset, srcSize, evenMask, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = partial_load<TypeParam>(values.begin() + offset, srcSize, evenMask, flag_convert);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Partial_Outside_Masked_Convert_DeducedVec, TypeParam, LoadableNativeTypes)
{
  using xvec::simd::partial_load;
  using xvec::simd::flag_convert;
  using value_type = typename TypeParam::value_type;

  constexpr auto srcSize = TypeParam::size - 1;
  constexpr auto offset = (TypeParam::size + 64) - srcSize;
  constexpr auto evenMask = getEvenMask<TypeParam>();
  static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(offset);

  static constexpr auto cexpected = [] {
    std::array<value_type, TypeParam::size> out{};
    for (std::size_t i = 0; i < srcSize; ++i)
      if ((i % 2) == 0)
        out[i] = static_cast<value_type>(cvalues[offset + i]);
    return out;
  }();

  auto values = cvalues;
  auto expected = cexpected;

  // Masked load from range.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load(crange, evenMask, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(offset);
    auto computedFromRange = partial_load(r, evenMask, flag_convert);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }

  // Use iterator pair.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load(cvalues.begin() + offset, cvalues.begin() + offset + srcSize, evenMask, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = partial_load(values.begin() + offset, values.begin() + offset + srcSize, evenMask, flag_convert);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  // Iterator + count
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load(cvalues.begin() + offset, srcSize, evenMask, flag_convert);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = partial_load(values.begin() + offset, srcSize, evenMask, flag_convert);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Partial_Outside_Unmasked_NoConvert_DeducedVec, TypeParam, LoadableNativeTypes)
{
  using xvec::simd::partial_load;
  using value_type = typename TypeParam::value_type;

  constexpr auto srcSize = TypeParam::size - 1;
  constexpr auto offset = (TypeParam::size + 64) - srcSize;
  static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(offset);

  static constexpr auto cexpected = [] {
    std::array<value_type, TypeParam::size> out{};
    for (std::size_t i = 0; i < srcSize; ++i)
      out[i] = static_cast<value_type>(cvalues[offset + i]);
    return out;
  }();

  auto values = cvalues;
  auto expected = cexpected;

  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load(cvalues.begin() + offset, cvalues.begin() + offset + srcSize);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = partial_load(values.begin() + offset, values.begin() + offset + srcSize);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }

  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load(cvalues.begin() + offset, srcSize);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = partial_load(values.begin() + offset, srcSize);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = partial_load(crange);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(offset);
    auto computedFromRange = partial_load(r);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_Partial_Outside_Masked_NoConvert_DeducedVec, TypeParam, LoadableNativeTypes)
{
  using xvec::simd::partial_load;
  using value_type = typename TypeParam::value_type;

  constexpr auto srcSize = TypeParam::size - 1;
  constexpr auto offset = (TypeParam::size + 64) - srcSize;
  constexpr auto evenMask = getEvenMask<TypeParam>();
  static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
  [[maybe_unused]] static constexpr auto crange = std::span{cvalues}.subspan(offset);

  static constexpr auto cexpected = [] {
    std::array<value_type, TypeParam::size> out{};
    for (std::size_t i = 0; i < srcSize; ++i)
      if ((i % 2) == 0)
        out[i] = static_cast<value_type>(cvalues[offset + i]);
    return out;
  }();

  auto values = cvalues;
  auto expected = cexpected;

  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load(crange, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    const auto r = std::span(values).subspan(offset);
    auto computedFromRange = partial_load(r, evenMask);
    BOOST_TEST(to_array(computedFromRange) == expected, boost::test_tools::per_element());
  }
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load(cvalues.begin() + offset, cvalues.begin() + offset + srcSize, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromPair = partial_load(values.begin() + offset, values.begin() + offset + srcSize, evenMask);
    BOOST_TEST(to_array(computedFromPair) == expected, boost::test_tools::per_element());
  }
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = partial_load(cvalues.begin() + offset, srcSize, evenMask);
      return to_array(v) == cexpected;
    }());
#endif

    auto computedFromCount = partial_load(values.begin() + offset, srcSize, evenMask);
    BOOST_TEST(to_array(computedFromCount) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Store_Inside_Unmasked_Convert, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::flag_convert;
  using value_type = typename TypeParam::value_type;

  static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
  static constexpr auto cexpected = with_iota_window(cvalues, 3, TypeParam::size);

  auto values = cvalues;
  auto expected = cexpected;

  // Range-specified output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, std::span(out.begin() + 3, TypeParam::size), flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, std::span(computed.begin() + 3, TypeParam::size), flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator+size output buffer
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, TypeParam::size, flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, TypeParam::size, flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator pair output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, out.begin() + 3 + TypeParam::size, flag_convert);
      return out == cexpected;
    }());
  #endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, computed.begin() + 3 + TypeParam::size, flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Store_Inside_Unmasked_Convert_ToDouble, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::flag_convert;

  static constexpr auto cvalues = getMemoryBlock<double, TypeParam::size + 64>();
  static constexpr auto cexpected = with_iota_window(cvalues, 3, TypeParam::size);

  auto values = cvalues;
  auto expected = cexpected;

  // Range-specified output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, std::span(out.begin() + 3, TypeParam::size), flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, std::span(computed.begin() + 3, TypeParam::size), flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator+size output buffer
  {
 #if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, TypeParam::size, flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, TypeParam::size, flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator pair output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, out.begin() + 3 + TypeParam::size, flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, computed.begin() + 3 + TypeParam::size, flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Store_Partial_Outside_Unmasked_NoConvert, TypeParam, LoadableSimdTypes)
{
  if constexpr (TypeParam::size < 3)
    return;

  using value_type = typename TypeParam::value_type;
  constexpr std::size_t destSize = TypeParam::size - 1;

  static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
  static constexpr auto cexpected = with_iota_window(cvalues, 3, destSize);

  auto values = cvalues;
  auto expected = cexpected;

  // Range-specified output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      partial_store(xvec::simd::iota<TypeParam>, std::span(out.begin() + 3, destSize));
      return out == cexpected;
    }());
#endif

    auto computed = values;
    partial_store(xvec::simd::iota<TypeParam>, std::span(computed.begin() + 3, destSize));
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator+size output buffer
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      partial_store(xvec::simd::iota<TypeParam>, out.begin() + 3, destSize);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    partial_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, destSize);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator pair output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      auto out = cvalues;
      partial_store(xvec::simd::iota<TypeParam>, out.begin() + 3, out.begin() + 3 + destSize);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    partial_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, computed.begin() + 3 + destSize);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Store_Inside_Masked_NoConvert, TypeParam, LoadableSimdTypes)
{
  using value_type = typename TypeParam::value_type;
  constexpr auto evenMask = getEvenMask<TypeParam>();

  static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
  static constexpr auto cexpected = with_iota_window_even(cvalues, 3, TypeParam::size);

  auto values = cvalues;
  auto expected = cexpected;

  // Range-specified output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, std::span(out.begin() + 3, TypeParam::size), evenMask);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, std::span(computed.begin() + 3, TypeParam::size), evenMask);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator+size output buffer
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, TypeParam::size, evenMask);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, TypeParam::size, evenMask);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator pair output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, out.begin() + 3 + TypeParam::size, evenMask);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, computed.begin() + 3 + TypeParam::size, evenMask);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Store_Inside_Masked_Convert_ToDouble, TypeParam, LoadableSimdTypes)
{
  using xvec::simd::flag_convert;
  constexpr auto evenMask = getEvenMask<TypeParam>();

  static constexpr auto cvalues = getMemoryBlock<double, TypeParam::size + 64>();
  static constexpr auto cexpected = with_iota_window_even(cvalues, 3, TypeParam::size);

  auto values = cvalues;
  auto expected = cexpected;

  // Range-specified output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, std::span(out.begin() + 3, TypeParam::size), evenMask, flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, std::span(computed.begin() + 3, TypeParam::size), evenMask, flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator+size output buffer
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, TypeParam::size, evenMask, flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, TypeParam::size, evenMask, flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }

  // Iterator pair output buffer.
  {
#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      auto out = cvalues;
      unchecked_store(xvec::simd::iota<TypeParam>, out.begin() + 3, out.begin() + 3 + TypeParam::size, evenMask, flag_convert);
      return out == cexpected;
    }());
#endif

    auto computed = values;
    unchecked_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, computed.begin() + 3 + TypeParam::size, evenMask, flag_convert);
    BOOST_TEST(computed == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Store_Partial_Outside_Masked_NoConvert, TypeParam, LoadableSimdTypes)
{
  using value_type = typename TypeParam::value_type;
  constexpr auto evenMask = getEvenMask<TypeParam>();

  if constexpr (TypeParam::size >= 3)
  {
    constexpr std::size_t destSize = TypeParam::size - 2;

    static constexpr auto cvalues = getMemoryBlock<value_type, TypeParam::size + 64>();
    static constexpr auto cexpected = with_iota_window_even(cvalues, 3, destSize);

    auto values = cvalues;
    auto expected = cexpected;

    // Range-specified output buffer.
    {
#if defined(_XVEC_HAS_CONSTEXPR)
      static_assert([=] {
        auto out = cvalues;
        partial_store(xvec::simd::iota<TypeParam>, std::span(out.begin() + 3, destSize), evenMask);
        return out == cexpected;
      }());
#endif

      auto computed = values;
      partial_store(xvec::simd::iota<TypeParam>, std::span(computed.begin() + 3, destSize), evenMask);
      BOOST_TEST(computed == expected, boost::test_tools::per_element());
    }

    // Iterator+size output buffer
    {
#if defined(_XVEC_HAS_CONSTEXPR)
      static_assert([=] {
        auto out = cvalues;
        partial_store(xvec::simd::iota<TypeParam>, out.begin() + 3, destSize, evenMask);
        return out == cexpected;
      }());
#endif

      auto computed = values;
      partial_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, destSize, evenMask);
      BOOST_TEST(computed == expected, boost::test_tools::per_element());
    }

    // Iterator pair output buffer.
    {
#if defined(_XVEC_HAS_CONSTEXPR)
      static_assert([=] {
        auto out = cvalues;
        partial_store(xvec::simd::iota<TypeParam>, out.begin() + 3, out.begin() + 3 + destSize, evenMask);
        return out == cexpected;
      }());
#endif

      auto computed = values;
      partial_store(xvec::simd::iota<TypeParam>, computed.begin() + 3, computed.begin() + 3 + destSize, evenMask);
      BOOST_TEST(computed == expected, boost::test_tools::per_element());
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Load_From_Static_Extent_Ranges, TypeParam, LoadableSimdTypes) {
  using value_type = typename TypeParam::value_type;
  constexpr std::size_t shortSize = TypeParam::size - 1;
  constexpr auto evenMask = getEvenMask<TypeParam>();

  auto exact = getMemoryBlock<value_type, TypeParam::size>();
  static_assert(xvec::simd::detail::span_extent_v<decltype(exact)> == TypeParam::size);
  BOOST_TEST(to_array(xvec::simd::partial_load<TypeParam>(exact)) == exact,
             boost::test_tools::per_element());
  BOOST_TEST(to_array(xvec::simd::unchecked_load<TypeParam>(exact)) == exact,
             boost::test_tools::per_element());

  auto maskedExpected = exact;
  for (std::size_t i = 1; i < maskedExpected.size(); i += 2)
    maskedExpected[i] = value_type{};
  BOOST_TEST(to_array(xvec::simd::partial_load<TypeParam>(exact, evenMask)) == maskedExpected,
             boost::test_tools::per_element());
  BOOST_TEST(to_array(xvec::simd::unchecked_load<TypeParam>(exact, evenMask)) == maskedExpected,
             boost::test_tools::per_element());

  auto larger = getMemoryBlock<value_type, TypeParam::size + 2>();
  std::array<value_type, TypeParam::size> largerExpected{};
  std::copy_n(larger.begin(), TypeParam::size, largerExpected.begin());
  BOOST_TEST(to_array(xvec::simd::partial_load<TypeParam>(larger)) == largerExpected,
             boost::test_tools::per_element());
  BOOST_TEST(to_array(xvec::simd::partial_load<TypeParam>(larger, evenMask)) == maskedExpected,
             boost::test_tools::per_element());
  BOOST_TEST(to_array(xvec::simd::unchecked_load<TypeParam>(larger, evenMask)) == maskedExpected,
             boost::test_tools::per_element());

  value_type carray[TypeParam::size] = {};
  std::iota(std::begin(carray), std::end(carray), value_type{});
  static_assert(xvec::simd::detail::span_extent_v<decltype(carray)> == TypeParam::size);
  const auto carrayExpected = getMemoryBlock<value_type, TypeParam::size>();
  BOOST_TEST(to_array(xvec::simd::unchecked_load<TypeParam>(carray)) == carrayExpected,
             boost::test_tools::per_element());

  auto shortArray = getMemoryBlock<value_type, shortSize>();
  std::span<value_type, shortSize> shortSpan(shortArray);
  static_assert(xvec::simd::detail::span_extent_v<decltype(shortSpan)> == shortSize);

  std::array<value_type, TypeParam::size> expected{};
  std::copy(shortArray.begin(), shortArray.end(), expected.begin());
  BOOST_TEST(to_array(xvec::simd::partial_load<TypeParam>(shortSpan)) == expected,
             boost::test_tools::per_element());

  for (std::size_t i = 1; i < expected.size(); i += 2)
    expected[i] = value_type{};
  BOOST_TEST(to_array(xvec::simd::partial_load<TypeParam>(shortSpan, evenMask)) == expected,
             boost::test_tools::per_element());

  const auto shortMask = xvec::simd::mask_from_count<TypeParam>(shortSize);
  std::copy(shortArray.begin(), shortArray.end(), expected.begin());
  BOOST_TEST(to_array(xvec::simd::unchecked_load<TypeParam>(shortSpan, shortMask)) == expected,
             boost::test_tools::per_element());
}

template <typename T, std::size_t Extent, std::size_t Padding>
struct StoreTest {
  static constexpr std::size_t storageSize = Extent + Padding;

  explicit StoreTest(T initialValue) {
    actual.fill(initialValue);
    expected.fill(initialValue);
  }

  std::span<T, Extent> destination() {
    return std::span<T, Extent>{actual.data(), Extent};
  }

  std::array<T, storageSize> actual{};
  std::array<T, storageSize> expected{};
};

BOOST_AUTO_TEST_CASE_TEMPLATE(Store_To_Static_Extent_Ranges, TypeParam,
                              LoadableSimdTypes) {
  using value_type = typename TypeParam::value_type;

  constexpr std::size_t vectorSize = TypeParam::size;
  constexpr std::size_t shortSize = vectorSize - 1;
  constexpr std::size_t largerSize = vectorSize + 2;
  constexpr value_type fillValue{42};

  constexpr auto evenMask = getEvenMask<TypeParam>();
  const auto values = xvec::simd::iota<TypeParam>;
  const auto expectedValues = to_array(values);

  { // Exact size match
    StoreTest<value_type, vectorSize, vectorSize> test{value_type{}};
    auto destination = test.destination();

    static_assert(xvec::simd::detail::span_extent_v<decltype(destination)> == vectorSize);

    xvec::simd::partial_store(values, destination);
    std::copy(expectedValues.begin(), expectedValues.end(),
              test.expected.begin());

    BOOST_TEST(test.actual == test.expected, boost::test_tools::per_element());
  }

  { // Masked exact
    StoreTest<value_type, vectorSize, vectorSize> test{fillValue};
    auto destination = test.destination();

    xvec::simd::partial_store(values, destination, evenMask);

    for (std::size_t i = 0; i < vectorSize; i += 2)
      test.expected[i] = expectedValues[i];

    BOOST_TEST(test.actual == test.expected,
               boost::test_tools::per_element());
  }

  { // Unchecked masked exact
    StoreTest<value_type, vectorSize, vectorSize> test{fillValue};
    auto destination = test.destination();

    xvec::simd::unchecked_store(values, destination, evenMask);

    for (std::size_t i = 0; i < vectorSize; i += 2)
      test.expected[i] = expectedValues[i];

    BOOST_TEST(test.actual == test.expected, boost::test_tools::per_element());
  }

  { // Larger destination
    StoreTest<value_type, largerSize, vectorSize> test{fillValue};
    auto destination = test.destination();

    static_assert(xvec::simd::detail::span_extent_v<decltype(destination)> == largerSize);

    xvec::simd::partial_store(values, destination);
    std::copy(expectedValues.begin(), expectedValues.end(),
              test.expected.begin());

    BOOST_TEST(test.actual == test.expected, boost::test_tools::per_element());
  }

  { // Masked larger destination
    StoreTest<value_type, largerSize, vectorSize> test{fillValue};
    auto destination = test.destination();

    xvec::simd::partial_store(values, destination, evenMask);

    for (std::size_t i = 0; i < vectorSize; i += 2)
      test.expected[i] = expectedValues[i];

    BOOST_TEST(test.actual == test.expected,
               boost::test_tools::per_element());
  }

  { // Unechedk masked larger destination
    StoreTest<value_type, largerSize, vectorSize> test{fillValue};
    auto destination = test.destination();

    xvec::simd::unchecked_store(values, destination, evenMask);

    for (std::size_t i = 0; i < vectorSize; i += 2)
      test.expected[i] = expectedValues[i];

    BOOST_TEST(test.actual == test.expected,
               boost::test_tools::per_element());
  }

  {
    StoreTest<value_type, vectorSize, vectorSize> test{value_type{}};
    auto destination = test.destination();

    xvec::simd::unchecked_store(values, destination);
    std::copy(expectedValues.begin(), expectedValues.end(),
              test.expected.begin());

    BOOST_TEST(test.actual == test.expected,
               boost::test_tools::per_element());
  }

  { // Smaller destination
    StoreTest<value_type, shortSize, vectorSize> test{value_type{}};
    auto destination = test.destination();

    static_assert(xvec::simd::detail::span_extent_v<decltype(destination)> == shortSize);
    xvec::simd::partial_store(values, destination);
    std::copy_n(expectedValues.begin(), shortSize, test.expected.begin());

    BOOST_TEST(test.actual == test.expected, boost::test_tools::per_element());
  }

  { // Smaller masked destination
    StoreTest<value_type, shortSize, vectorSize> test{fillValue};
    auto destination = test.destination();

    xvec::simd::partial_store(values, destination, evenMask);

    for (std::size_t i = 0; i < shortSize; i += 2)
      test.expected[i] = expectedValues[i];

    BOOST_TEST(test.actual == test.expected, boost::test_tools::per_element());
  }

  { // Unchecked smaller destination
    StoreTest<value_type, shortSize, vectorSize> test{fillValue};
    auto destination = test.destination();

    xvec::simd::unchecked_store(values, destination, xvec::simd::mask_from_count<TypeParam>(shortSize));

    std::copy_n(expectedValues.begin(), shortSize, test.expected.begin());

    BOOST_TEST(test.actual == test.expected, boost::test_tools::per_element());
  }

}
