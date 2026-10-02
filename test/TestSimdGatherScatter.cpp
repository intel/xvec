//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

// :TODO: The tests in this file assume that the values are loaded from real-valued, so
// what should happen when a complex value is loaded from a real-valued source.
// Should it convert, or break? For the moment it breaks.
using LoadableSimdTypes = ArithmeticSimdTypes;

// Build a memory to use for gather/scatter with pre-defined values.
template<typename _Tp, std::size_t _MemSize>
constexpr std::array<_Tp, _MemSize> build_memory() {
  std::array<_Tp, _MemSize> memTable{};
  for (std::size_t i = 0; i < _MemSize; ++i)
  {
    // Start filling from a curious number so that it doesn't get confused for an index.
    memTable[i] = static_cast<_Tp>(i + 1947);
  }
  return memTable;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GatherFromContiguousRange, TypeParam, AllSimdTypes)
{
  using xvec::rebind_cast;

  using _Tp = typename TypeParam::value_type;

  // Create a backing allocation containing a smaller readable range.
  constexpr int memorySize = 256;
  const auto memTable = build_memory<_Tp, memorySize>();

  constexpr int tableOffset = 44; // Allow space before the range for negative indexes.
  constexpr int rangeSize = 100;
  constexpr int maximumIndex = 127;
  const auto table = std::ranges::subrange(memTable.begin() + tableOffset, memTable.begin() + tableOffset + rangeSize);

  // Most random indexes are valid, while values from 100 through 127 exercise
  // the upper bound. Every value remains representable by a signed byte.
  auto indexes = rebind_cast<short>(GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(maximumIndex));
  indexes = set_element(indexes, 0, -2);

  // Generate the expected values for every index width.
  const auto checkedExpected = applyUnary(indexes, [=](auto idx) {
    return idx >= 0 && idx < rangeSize ? table[idx] : _Tp{};
  });
  const auto uncheckedExpected = applyUnary(indexes, [=](auto idx) {
    return memTable[tableOffset + idx];
  });

  // Gather using four different signed index sizes.
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<int8_t>(indexes)), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<int16_t>(indexes)), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<int32_t>(indexes)), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<int64_t>(indexes)), checkedExpected);

  // The unchecked variant may read outside the logical range, but every index
  // still addresses valid storage in the backing allocation.
  BOOST_SIMD_EQUAL(unchecked_gather_from(table, rebind_cast<int64_t>(indexes)), uncheckedExpected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GatherFromContiguousRangeWithConversion, TypeParam, LoadableSimdTypes)
{
  using xvec::rebind_cast;

  using _Tp = typename TypeParam::value_type;

  // Create a backing allocation of bytes containing a smaller readable range.
  constexpr int memorySize = 256;
  const auto memTable = build_memory<int8_t, memorySize>();

  constexpr int tableOffset = 44; // Allow space before the range for negative indexes.
  constexpr int rangeSize = 100;
  constexpr int maximumIndex = 127;
  const auto table = std::ranges::subrange(memTable.begin() + tableOffset, memTable.begin() + tableOffset + rangeSize);

  // Most random indexes are valid, while values from 100 through 127 exercise
  // the upper bound. Every value remains representable by a signed byte.
  auto indexes = rebind_cast<short>(GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(maximumIndex));
  indexes = set_element(indexes, 0, -2);

  // Generate the expected converted values for every index width.
  const auto checkedExpected = applyUnary(indexes, [=](auto idx) -> _Tp {
    return idx >= 0 && idx < rangeSize ? table[idx] : _Tp{};
  });
  const auto uncheckedExpected = applyUnary(indexes, [=](auto idx) -> _Tp {
    return memTable[tableOffset + idx];
  });

  // Gather using four different signed index sizes.
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<int8_t>(indexes), xvec::simd::flag_convert), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<int16_t>(indexes), xvec::simd::flag_convert), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<int32_t>(indexes), xvec::simd::flag_convert), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<int64_t>(indexes), xvec::simd::flag_convert), checkedExpected);

  // The unchecked variant may read outside the logical range, but every index
  // still addresses valid storage in the backing allocation.
  BOOST_SIMD_EQUAL(unchecked_gather_from<TypeParam>(table, rebind_cast<int64_t>(indexes), xvec::simd::flag_convert), uncheckedExpected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskedGatherFromContiguousRange, TypeParam, AllSimdTypes)
{
  using xvec::rebind_cast;

  using _Tp = typename TypeParam::value_type;

  // Create a backing allocation containing a smaller readable range.
  constexpr int memorySize = 256;
  const auto memTable = build_memory<_Tp, memorySize>();

  constexpr int tableOffset = 44; // Allow space before the range for negative indexes.
  constexpr int rangeSize = 100;
  constexpr int maximumIndex = 127;
  const auto table = std::ranges::subrange(memTable.begin() + tableOffset, memTable.begin() + tableOffset + rangeSize);

  // Most random indexes are valid, while values from 100 through 127 exercise
  // the upper bound. Every value remains representable by a signed byte.
  auto indexes = rebind_cast<short>(GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(maximumIndex));
  indexes = set_element(indexes, 0, -2);

  // Create a random mask and ensure the negative index is enabled.
  auto maskBits = getRandomBitset<TypeParam::size>();
  maskBits[0] = true;
  const auto mask = xvec::simd::mask<unsigned short, TypeParam::size>(maskBits);

  // Generate the expected values for every index width.
  const auto expected = TypeParam([=](auto i) {
    return maskBits[i] && indexes[i] >= 0 && indexes[i] < rangeSize ? table[indexes[i]] : _Tp{};
  });

  // Masked gather using four different signed index sizes.
  const auto indexesAs8 = rebind_cast<int8_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs8)::mask_type(mask), indexesAs8), expected);

  const auto indexesAs16 = rebind_cast<int16_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs16)::mask_type(mask), indexesAs16), expected);

  const auto indexesAs32 = rebind_cast<int32_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs32)::mask_type(mask), indexesAs32), expected);

  const auto indexesAs64 = rebind_cast<int64_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs64)::mask_type(mask), indexesAs64), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskedGatherFromContiguousRangeWithConversion, TypeParam, LoadableSimdTypes)
{
  using xvec::rebind_cast;

  using _Tp = typename TypeParam::value_type;

  // Create a backing allocation of bytes containing a smaller readable range.
  constexpr int memorySize = 256;
  const auto memTable = build_memory<int8_t, memorySize>();

  constexpr int tableOffset = 44; // Allow space before the range for negative indexes.
  constexpr int rangeSize = 100;
  constexpr int maximumIndex = 127;
  const auto table = std::ranges::subrange(memTable.begin() + tableOffset, memTable.begin() + tableOffset + rangeSize);

  // Most random indexes are valid, while values from 100 through 127 exercise
  // the upper bound. Every value remains representable by a signed byte.
  auto indexes = rebind_cast<short>(GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(maximumIndex));
  indexes = set_element(indexes, 0, -2);

  // Create a random mask and ensure the negative index is enabled.
  auto maskBits = getRandomBitset<TypeParam::size>();
  maskBits[0] = true;
  const auto mask = xvec::simd::mask<unsigned short, TypeParam::size>(maskBits);

  // Generate the expected converted values for every index width.
  const auto checkedExpected = TypeParam([=](auto i) -> _Tp {
    return maskBits[i] && indexes[i] >= 0 && indexes[i] < rangeSize ? table[indexes[i]] : _Tp{};
  });
  const auto uncheckedExpected = TypeParam([=](auto i) -> _Tp {
    return maskBits[i] ? memTable[tableOffset + indexes[i]] : _Tp{};
  });

  // Masked gather using four different signed index sizes.
  const auto indexesAs8 = rebind_cast<int8_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs8)::mask_type(mask), indexesAs8, xvec::simd::flag_convert), checkedExpected);

  const auto indexesAs16 = rebind_cast<int16_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs16)::mask_type(mask), indexesAs16, xvec::simd::flag_convert), checkedExpected);

  const auto indexesAs32 = rebind_cast<int32_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs32)::mask_type(mask), indexesAs32, xvec::simd::flag_convert), checkedExpected);

  const auto indexesAs64 = rebind_cast<int64_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs64)::mask_type(mask), indexesAs64, xvec::simd::flag_convert), checkedExpected);

  // The unchecked variant may read outside the logical range, but every index
  // still addresses valid storage in the backing allocation.
  BOOST_SIMD_EQUAL(unchecked_gather_from<TypeParam>(table, typename decltype(indexesAs64)::mask_type(mask), indexesAs64, xvec::simd::flag_convert), uncheckedExpected);
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprGatherFromContiguousRange, TypeParam, LoadableSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  // Create a contiguous lookup table from bytes, which can be converted to all the other types.
  constexpr int memorySize = 200;
  constexpr auto table = build_memory<int8_t,memorySize>();

  // Create a set of random indexes and masks into the memory from which to read.
  using SimdIndex = xvec::simd::vec<unsigned short, TypeParam::size>;
  constexpr auto indexes = GetConstexprRandomVector<SimdIndex, 0>(memorySize);
  constexpr auto mask = GetConstexprRandomVector<typename SimdIndex::mask_type, 1>();

  constexpr auto expectedUnmasked = TypeParam([=](auto i) -> _Tp { return table[indexes[i]]; });
  constexpr auto expectedMasked = TypeParam([=](auto i) -> _Tp { return (mask[i]) ? table[indexes[i]] : _Tp{}; });

  // Some constexpr gathers. Not exhaustive as they mostly check that constexpr is callable.
  constexpr auto ceExpUnmasked = partial_gather_from<TypeParam>(table, indexes, xvec::simd::flag_convert);
  constexpr auto ceExpMasked = partial_gather_from<TypeParam>(table, mask, indexes, xvec::simd::flag_convert);

  static_assert(to_array(ceExpUnmasked) == to_array(expectedUnmasked));
  static_assert(to_array(ceExpMasked) == to_array(expectedMasked));
}
#endif // _XVEC_HAS_CONSTEXPR

BOOST_AUTO_TEST_CASE_TEMPLATE(ScatterToRangeWithSameType, TypeParam, AllSimdTypes)
{
  using xvec::rebind_cast;

  SimdTestFixture<TypeParam> f;

  // Create a backing allocation containing a smaller writable range. Comparing
  // the entire allocation verifies that out-of-range writes are rejected.
  constexpr int memorySize = 256;
  const auto originalMemory = build_memory<typename TypeParam::value_type, memorySize>();

  constexpr int tableOffset = 44; // Allow space before the range for negative indexes.
  constexpr int rangeSize = 100;
  constexpr int maximumIndex = 127;

  // Most random indexes are valid, while values from 100 through 127 exercise
  // the upper bound. Every value remains representable by a signed byte.
  auto indexes = rebind_cast<short>(GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(maximumIndex));
  indexes = set_element(indexes, 0, -2);

  // Compute the expected memory after scattering into it.
  auto expectedMemory = originalMemory;
  for (int i = 0; i < TypeParam::size; ++i)
    if (indexes[i] >= 0 && indexes[i] < rangeSize)
      expectedMemory[tableOffset + indexes[i]] = f.v0[i];

  // Scatter using 8-bit signed indexes.
  auto memory8 = originalMemory;
  auto table8 = std::ranges::subrange(memory8.begin() + tableOffset, memory8.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table8, rebind_cast<int8_t>(indexes));
  BOOST_TEST(memory8 == expectedMemory, boost::test_tools::per_element());

  // Scatter using 16-bit signed indexes.
  auto memory16 = originalMemory;
  auto table16 = std::ranges::subrange(memory16.begin() + tableOffset, memory16.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table16, rebind_cast<int16_t>(indexes));
  BOOST_TEST(memory16 == expectedMemory, boost::test_tools::per_element());

  // Scatter using 32-bit signed indexes.
  auto memory32 = originalMemory;
  auto table32 = std::ranges::subrange(memory32.begin() + tableOffset, memory32.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table32, rebind_cast<int32_t>(indexes));
  BOOST_TEST(memory32 == expectedMemory, boost::test_tools::per_element());

  // Scatter using 64-bit signed indexes.
  auto memory64 = originalMemory;
  auto table64 = std::ranges::subrange(memory64.begin() + tableOffset, memory64.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table64, rebind_cast<int64_t>(indexes));
  BOOST_TEST(memory64 == expectedMemory, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskedScatterToContiguousRange, TypeParam, AllSimdTypes)
{
  using xvec::rebind_cast;

  SimdTestFixture<TypeParam> f;

  // Create a backing allocation containing a smaller writable range. Comparing
  // the entire allocation verifies that out-of-range writes are rejected.
  constexpr int memorySize = 256;
  const auto originalMemory = build_memory<typename TypeParam::value_type, memorySize>();

  constexpr int tableOffset = 44; // Allow space before the range for negative indexes.
  constexpr int rangeSize = 100;
  constexpr int maximumIndex = 127;

  // Most random indexes are valid, while values from 100 through 127 exercise
  // the upper bound. Every value remains representable by a signed byte.
  auto indexes = rebind_cast<short>(GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(maximumIndex));
  indexes = set_element(indexes, 0, -2);

  // Create a random mask and ensure the negative index is enabled.
  auto maskBits = getRandomBitset<TypeParam::size>();
  maskBits[0] = true;
  const auto mask = xvec::simd::mask<unsigned short, TypeParam::size>(maskBits);

  // Compute the expected memory for every index width.
  auto expectedMemory = originalMemory;
  for (std::size_t i = 0; i < TypeParam::size; ++i)
    if (maskBits[i] && indexes[i] >= 0 && indexes[i] < rangeSize)
      expectedMemory[tableOffset + indexes[i]] = f.v0[i];

  // Scatter using 8-bit signed indexes.
  const auto indexesAs8 = rebind_cast<int8_t>(indexes);
  auto memory8 = originalMemory;
  auto table8 = std::ranges::subrange(memory8.begin() + tableOffset, memory8.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table8, typename decltype(indexesAs8)::mask_type(mask), indexesAs8);
  BOOST_TEST(memory8 == expectedMemory, boost::test_tools::per_element());

  // Scatter using 16-bit signed indexes.
  const auto indexesAs16 = rebind_cast<int16_t>(indexes);
  auto memory16 = originalMemory;
  auto table16 = std::ranges::subrange(memory16.begin() + tableOffset, memory16.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table16, typename decltype(indexesAs16)::mask_type(mask), indexesAs16);
  BOOST_TEST(memory16 == expectedMemory, boost::test_tools::per_element());

  // Scatter using 32-bit signed indexes.
  const auto indexesAs32 = rebind_cast<int32_t>(indexes);
  auto memory32 = originalMemory;
  auto table32 = std::ranges::subrange(memory32.begin() + tableOffset, memory32.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table32, typename decltype(indexesAs32)::mask_type(mask), indexesAs32);
  BOOST_TEST(memory32 == expectedMemory, boost::test_tools::per_element());

  // Scatter using 64-bit signed indexes.
  const auto indexesAs64 = rebind_cast<int64_t>(indexes);
  auto memory64 = originalMemory;
  auto table64 = std::ranges::subrange(memory64.begin() + tableOffset, memory64.begin() + tableOffset + rangeSize);
  partial_scatter_to(f.v0, table64, typename decltype(indexesAs64)::mask_type(mask), indexesAs64);
  BOOST_TEST(memory64 == expectedMemory, boost::test_tools::per_element());
}

#if defined(_XVEC_HAS_CONSTEXPR)

/// Create a function which generates a constexpr memory using scatter. iota is
/// written to the indexes.
template<xvec::simd::vec_type _S>
constexpr auto constexprMemory(auto memory, const _S& indexes)
{
  auto localMemory = memory;
  partial_scatter_to(xvec::simd::iota<_S>, localMemory, indexes, xvec::simd::flag_convert);
  return localMemory;
}

template<xvec::simd::vec_type _S>
constexpr auto constexprMemory(auto memory, const _S& indexes, const typename _S::mask_type& m)
{
  auto localMemory = memory;
  partial_scatter_to(xvec::simd::iota<_S>, localMemory, m, indexes, xvec::simd::flag_convert);
  return localMemory;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprScatterToContiguousRange, TypeParam, ArithmeticSimdTypes)
{
  constexpr int memorySize = 200;
  constexpr auto originalMemory = build_memory<typename TypeParam::value_type, memorySize>();

  constexpr auto mask = GetConstexprRandomVector<xvec::simd::mask<unsigned short, TypeParam::size>, 0>();
  constexpr auto indexes = GetConstexprRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>, 1>(memorySize);

  constexpr auto expectedUnmaskedMemory = [=] {
    auto out = originalMemory;
    for (std::size_t i = 0; i < TypeParam::size; ++i)
      out[indexes[i]] = static_cast<typename TypeParam::value_type>(i);
    return out;
  }();

  constexpr auto expectedMaskedMemory = [=] {
    auto out = originalMemory;
    for (std::size_t i = 0; i < TypeParam::size; ++i)
      if (mask[i])
        out[indexes[i]] = static_cast<typename TypeParam::value_type>(i);
    return out;
  }();

  // Create a set of random indexes into the memory to which to write.
  constexpr auto computedUnmaskedMemory = constexprMemory(originalMemory, indexes);
  constexpr auto computedMaskedMemory = constexprMemory(originalMemory, indexes, mask);

  static_assert(computedUnmaskedMemory == expectedUnmaskedMemory);
  static_assert(computedMaskedMemory == expectedMaskedMemory);
}
#endif
