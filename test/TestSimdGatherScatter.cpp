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
  
  // Create a contiguous lookup table, and a range into that memory which is
  // half the size. The gather should only permit values in the range to be
  // read, not those in the whole table.
  constexpr int memorySize = 1024;
  const auto memTable = build_memory<typename TypeParam::value_type, memorySize>();
  constexpr int rangeSize = 200;
  const auto table = std::ranges::subrange(memTable.begin(), memTable.begin() + rangeSize);

  // Create a set of random indexes into the entire memory table, not just the smaller range.
  const auto indexes = GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(memorySize - 1);

  const auto checkedExpected = applyUnary(indexes, [=](auto idx) { return idx < rangeSize ? memTable[idx] : typename TypeParam::value_type(); });
  const auto uncheckedExpected = applyUnary(indexes, [=](auto idx) { return memTable[idx]; });
  const auto expectedByteIndexed = applyUnary(indexes, [=](auto idx) { return (idx & 0xff) < rangeSize ? memTable[idx & 0xff] : typename TypeParam::value_type(); });

  // Gather using 4 different index sizes.
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<uint8_t>(indexes)), expectedByteIndexed);
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<uint16_t>(indexes)), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<uint32_t>(indexes)), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from(table, rebind_cast<uint64_t>(indexes)), checkedExpected);

  // One unchecked variant - it uses the same pathway through the functions, just with a different check flag.
  BOOST_SIMD_EQUAL(unchecked_gather_from(table, rebind_cast<uint64_t>(indexes)), uncheckedExpected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GatherFromContiguousRangeWithConversion, TypeParam, LoadableSimdTypes)
{
  using xvec::rebind_cast;

  using _Tp = typename TypeParam::value_type;

  // Create a contiguous lookup table of bytes (since they can be converted into
  // any other loadable type), and a range into that memory which is half the
  // size. The gather should only permit values in the range to be read, not
  // those in the whole table.
  constexpr int memorySize = 1024;
  const auto memTable = build_memory<int8_t, memorySize>();
  constexpr int rangeSize = 200;
  const auto table = std::ranges::subrange(memTable.begin(), memTable.begin() + rangeSize);

  // Create a set of random indexes into the entire memory table, not just the smaller range.
  const auto indexes = GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(memorySize - 1);

  const auto checkedExpected = applyUnary(indexes, [=](auto idx) -> _Tp { return idx < rangeSize ? memTable[idx] : _Tp(); });
  const auto uncheckedExpected = applyUnary(indexes, [=](auto idx) -> _Tp { return memTable[idx]; });
  const auto expectedByteIndexed = applyUnary(indexes, [=](auto idx) -> _Tp { return (idx & 0xff) < rangeSize ? memTable[idx & 0xff] : _Tp(); });

  // Gather using 4 different index sizes.
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<uint8_t>(indexes), xvec::simd::flag_convert), expectedByteIndexed);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<uint16_t>(indexes), xvec::simd::flag_convert), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<uint32_t>(indexes), xvec::simd::flag_convert), checkedExpected);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, rebind_cast<uint64_t>(indexes), xvec::simd::flag_convert), checkedExpected);

  // One unchecked variant - it uses the same pathway through the functions, just with a different check flag.
  BOOST_SIMD_EQUAL(unchecked_gather_from<TypeParam>(table, rebind_cast<uint64_t>(indexes), xvec::simd::flag_convert), uncheckedExpected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskedGatherFromContiguousRange, TypeParam, AllSimdTypes)
{
  using xvec::rebind_cast;

  // Create a contiguous lookup table.
  constexpr int memorySize = 1024;
  const auto memTable = build_memory<typename TypeParam::value_type, memorySize>();
  constexpr int rangeSize = 200;
  const auto table = std::ranges::subrange(memTable.begin(), memTable.begin() + rangeSize);

  // Create a set of random indexes into the memory from which to read.
  const auto indexes = GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(memorySize - 1);

  // Create a random mask.
  auto maskBits = getRandomBitset<TypeParam::size>();
  auto mask = xvec::simd::mask<unsigned short, TypeParam::size>(maskBits);

  auto expected = TypeParam([=](auto i) {
    return (maskBits[i] && indexes[i] < rangeSize) ? table[indexes[i]] : typename TypeParam::value_type{};
  });
  auto expectedByteIndexed = TypeParam([=](auto i) {
    return maskBits[i] && ((indexes[i] & 0xFF) < rangeSize) ? table[indexes[i] & 0xFF] : typename TypeParam::value_type{};
  });

  auto indexesAs8 = rebind_cast<uint8_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs8)::mask_type(mask), indexesAs8), expectedByteIndexed);

  auto indexesAs16 = rebind_cast<uint16_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs16)::mask_type(mask), indexesAs16), expected);

  auto indexesAs32 = rebind_cast<uint32_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs32)::mask_type(mask), indexesAs32), expected);

  auto indexesAs64 = rebind_cast<uint64_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from(table, typename decltype(indexesAs64)::mask_type(mask), indexesAs64), expected);

}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskedGatherFromContiguousRangeWithConversion, TypeParam, LoadableSimdTypes)
{
  using xvec::rebind_cast;

  using _Tp = typename TypeParam::value_type;

  // Create a contiguous lookup table from bytes, which can be converted to all the other types.
  constexpr int memorySize = 1024;
  const auto memTable = build_memory<int8_t, memorySize>();
  constexpr int rangeSize = 200;
  const auto table = std::ranges::subrange(memTable.begin(), memTable.begin() + rangeSize);

  // Create a set of random indexes into the memory from which to read.
  const auto indexes = GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(memorySize - 1);

  // Create a random mask.
  auto maskBits = getRandomBitset<TypeParam::size>();
  auto mask = xvec::simd::mask<unsigned short, TypeParam::size>(maskBits);

  auto checkedExpected = TypeParam([=](auto i) -> _Tp {
    return (maskBits[i] && indexes[i] < rangeSize) ? table[indexes[i]] : _Tp{};
  });
  auto uncheckedExpected = TypeParam([=](auto i) -> _Tp {
    return (maskBits[i]) ? table[indexes[i]] : _Tp{};
  });
  auto expectedByteIndexed = TypeParam([=](auto i) -> _Tp{
    return maskBits[i] && ((indexes[i] & 0xFF) < rangeSize) ? table[indexes[i] & 0xFF] : _Tp{};
  });

  auto indexesAs8 = rebind_cast<uint8_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs8)::mask_type(mask), indexesAs8, xvec::simd::flag_convert), expectedByteIndexed);

  auto indexesAs16 = rebind_cast<uint16_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs16)::mask_type(mask), indexesAs16, xvec::simd::flag_convert), checkedExpected);

  auto indexesAs32 = rebind_cast<uint32_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs32)::mask_type(mask), indexesAs32, xvec::simd::flag_convert), checkedExpected);

  auto indexesAs64 = rebind_cast<uint64_t>(indexes);
  BOOST_SIMD_EQUAL(partial_gather_from<TypeParam>(table, typename decltype(indexesAs64)::mask_type(mask), indexesAs64, xvec::simd::flag_convert), checkedExpected);
  
  // One unchecked variant - it uses the same pathway through the functions, just with a different check flag.
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
  SimdTestFixture<TypeParam> f;

  // Create a contiguous table to write into. Only part of that table will be
  // written, to check that the bounds checks work.
  constexpr int memorySize = 1024;
  const auto originalMemory = build_memory<typename TypeParam::value_type, memorySize>();
  constexpr int rangeSize = 200;

  // Create a set of random indexes into the memory to which to write.
  const auto indexes = GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(memorySize - 1);

  // Compute the expected memory after scattering into it.
  auto expectedMemory = originalMemory;
  for (int i=0; i<TypeParam::size; ++i)
    if (indexes[i] < rangeSize)
      expectedMemory[indexes[i]] = f.v0[i];

  auto computedMemory = originalMemory;
  auto table = std::ranges::subrange(computedMemory.begin(), computedMemory.begin() + rangeSize);
  partial_scatter_to(f.v0, table, indexes);
  BOOST_TEST(computedMemory == expectedMemory, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MaskedScatterToContiguousRange, TypeParam, AllSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // Create a contiguous table to write into. Only part of that table will be
  // written, to check that the bounds checks work.
  constexpr int memorySize = 1024;
  const auto originalMemory = build_memory<typename TypeParam::value_type, memorySize>();
  constexpr int rangeSize = 200;

  // Create a set of random indexes into the memory to which to write.
  const auto indexes = GetRandomVector<xvec::simd::vec<unsigned short, TypeParam::size>>(memorySize - 1);

  // Create a random mask.
  auto maskBits = getRandomBitset<TypeParam::size>();
  auto mask = xvec::simd::mask<unsigned short, TypeParam::size>(maskBits);

  // Compute the expected memory after scattering into unmasked locations.
  auto expectedMemory = originalMemory;
  for (int i=0; i<TypeParam::size; ++i)
    if (indexes[i] < rangeSize && maskBits[i])
      expectedMemory[indexes[i]] = f.v0[i];

  auto computedMemory = originalMemory;
  auto table = std::ranges::subrange(computedMemory.begin(), computedMemory.begin() + rangeSize);

  partial_scatter_to(f.v0, table, mask, indexes);
  BOOST_TEST(computedMemory == expectedMemory, boost::test_tools::per_element());
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
