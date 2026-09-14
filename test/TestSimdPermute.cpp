//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(ZeroingPermute, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  std::array<typename TypeParam::value_type, TypeParam::size> expected{};
  for (std::size_t i = 0; i < TypeParam::size; ++i)
    expected[i] = (i % 3 == 0) ? typename TypeParam::value_type{} : f.v0[i];

  const auto computed = permute(f.v0, [](auto idx) { return (idx % 3) == 0 ? xvec::simd::zero_element : idx; });
  BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  static_assert([] {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 11>();
    constexpr auto c = permute(v, [](auto idx) { return (idx % 3) == 0 ? xvec::simd::zero_element : idx; });

    std::array<typename TypeParam::value_type, TypeParam::size> e{};
    for (std::size_t i = 0; i < TypeParam::size; ++i)
      e[i] = (i % 3 == 0) ? typename TypeParam::value_type{} : v[i];

    return to_array(c) == e;
  }());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(UninitElementPermute, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  // Generator: return uninit_element for positions where idx % 3 == 1, else return idx.
  // Positions where idx % 3 != 1 must equal the corresponding input element.
  const auto computed = permute(f.v0, [](auto idx) {
    return (idx % 3 == 1) ? xvec::simd::uninit_element : idx;
  });
  for (std::size_t i = 0; i < TypeParam::size; ++i)
  {
    if (i % 3 != 1)
      BOOST_TEST(computed[i] == f.v0[i]);
  }

#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr auto v = GetConstexprRandomVector<TypeParam, 15>();
  constexpr auto c = permute(v, [](auto idx) {
    return (idx % 3 == 1) ? xvec::simd::uninit_element : idx;
  });

  static_assert([=] {
    for (std::size_t i = 0; i < TypeParam::size; ++i)
    {
      if (i % 3 != 1 && c[i] != v[i])
        return false;
    }
    return true;
  }());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FunctionPermuteEven, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() >= 4)
  {
    auto even = [](std::size_t i) -> std::size_t { return i * 2; };

    std::array<typename TypeParam::value_type, TypeParam::size() / 2> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = f.v0[i * 2];

    const auto computed = xvec::simd::permute<TypeParam::size() / 2>(f.v0, even);
    BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = GetConstexprRandomVector<TypeParam, 12>();
      constexpr auto c = xvec::simd::permute<TypeParam::size() / 2>(v, [](std::size_t i) { return i * 2; });

      std::array<typename TypeParam::value_type, TypeParam::size() / 2> e{};
      for (std::size_t i = 0; i < e.size(); ++i)
        e[i] = v[i * 2];

      return to_array(c) == e;
    }());
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FunctionPermuteReverse, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() >= 4)
  {
    std::array<typename TypeParam::value_type, TypeParam::size()> expected{};
    for (std::size_t i = 0; i < TypeParam::size(); ++i)
      expected[i] = f.v0[(TypeParam::size() - 1) - i];

    auto reverse = [](std::size_t i) -> std::size_t { return (TypeParam::size() - 1) - i; };
    const auto computed = permute(f.v0, reverse);

    BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      constexpr auto v = GetConstexprRandomVector<TypeParam, 13>();
      constexpr auto c = permute(v, [](std::size_t i) { return (TypeParam::size() - 1) - i; });

      std::array<typename TypeParam::value_type, TypeParam::size()> e{};
      for (std::size_t i = 0; i < TypeParam::size(); ++i)
        e[i] = v[(TypeParam::size() - 1) - i];

      return to_array(c) == e;
    }());
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReverseGeneratedPermuteUsingSize, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  std::array<typename TypeParam::value_type, TypeParam::size> expected{};
  for (std::size_t i = 0; i < TypeParam::size; ++i)
    expected[i] = f.v0[(TypeParam::size - 1) - i];

  const auto computed = permute(f.v0, [](auto idx, auto size) { return (size - 1) - idx; });
  BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  static_assert([] {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 14>();
    constexpr auto c = permute(v, [](auto idx, auto size) { return (size - 1) - idx; });

    std::array<typename TypeParam::value_type, TypeParam::size> e{};
    for (std::size_t i = 0; i < TypeParam::size; ++i)
      e[i] = v[(TypeParam::size - 1) - i];

    return to_array(c) == e;
  }());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BySmallIndex, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() >= 4)
  {
    using UnsignedIndexSimd = xvec::simd::vec<unsigned short, 5>;
    const auto unsignedIndexes = GetRandomVector<UnsignedIndexSimd>(TypeParam::size() - 1);
    const auto signedIndexes = xvec::rebind_cast<signed short>(unsignedIndexes);

    std::array<typename TypeParam::value_type, 5> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = f.v0[unsignedIndexes[i]];

    const auto computedOptionUnsignedA = permute(f.v0, unsignedIndexes);
    BOOST_TEST(to_array(computedOptionUnsignedA) == expected, boost::test_tools::per_element());

    const auto computedOptionSignedA = permute(f.v0, signedIndexes);
    BOOST_TEST(to_array(computedOptionSignedA) == expected, boost::test_tools::per_element());

    const auto computedOptionUnsignedB = f.v0[unsignedIndexes];
    BOOST_TEST(to_array(computedOptionUnsignedB) == expected, boost::test_tools::per_element());

    const auto computedOptionSignedB = f.v0[signedIndexes];
    BOOST_TEST(to_array(computedOptionSignedB) == expected, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([] {
      using UnsignedIndexSimdCE = xvec::simd::vec<unsigned short, 5>;
      constexpr auto ceD = GetConstexprRandomVector<TypeParam, 0>();
      constexpr auto ceI = GetConstexprRandomVector<UnsignedIndexSimdCE, 1>(TypeParam::size() - 1);

      constexpr auto ceCompF = permute(ceD, ceI);
      constexpr auto ceCompS = ceD[ceI];

      std::array<typename TypeParam::value_type, 5> e{};
      for (std::size_t i = 0; i < e.size(); ++i)
        e[i] = ceD[ceI[i]];

      return (to_array(ceCompF) == e) && (to_array(ceCompS) == e);
    }());
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ByBigIndex, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() >= 4)
  {
    constexpr int numIndexes = 67;
    const auto indexes = GetRandomVector<xvec::simd::vec<unsigned short, numIndexes>>(TypeParam::size() - 1);

    std::array<typename TypeParam::value_type, numIndexes> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = f.v0[indexes[i]];

    const auto computedOptionA = permute(f.v0, indexes);
    BOOST_TEST(to_array(computedOptionA) == expected, boost::test_tools::per_element());

    const auto computedOptionB = f.v0[indexes];
    BOOST_TEST(to_array(computedOptionB) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SplitEvenlyInTwoBySize, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() > 2)
  {
    constexpr auto lsize = (TypeParam::size() + 1) / 2;
    const auto original = to_array(f.v0);

    std::array<typename TypeParam::value_type, lsize> lowerExpected{};
    for (std::size_t i = 0; i < lsize; ++i)
      lowerExpected[i] = original[i];

    std::array<typename TypeParam::value_type, TypeParam::size() - lsize> upperExpected{};
    for (std::size_t i = 0; i < upperExpected.size(); ++i)
      upperExpected[i] = original[lsize + i];

    const auto [l, u] = xvec::simd::chunk<lsize>(f.v0);
    BOOST_TEST(to_array(l) == lowerExpected, boost::test_tools::per_element());
    BOOST_TEST(to_array(u) == upperExpected, boost::test_tools::per_element());
    BOOST_TEST(to_array(cat(l, u)) == original, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = GetConstexprRandomVector<TypeParam, 21>();
      constexpr auto p = xvec::simd::chunk<lsize>(v);
      constexpr auto lce = std::get<0>(p);
      constexpr auto uce = std::get<1>(p);

      std::array<typename TypeParam::value_type, lsize> le{};
      for (std::size_t i = 0; i < lsize; ++i) le[i] = v[i];

      std::array<typename TypeParam::value_type, TypeParam::size() - lsize> ue{};
      for (std::size_t i = 0; i < ue.size(); ++i) ue[i] = v[lsize + i];

      return (to_array(lce) == le) &&
             (to_array(uce) == ue) &&
             (to_array(cat(lce, uce)) == to_array(v));
    }());
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SplitEvenlyInTwoByType, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() > 2)
  {
    constexpr auto lsize = (TypeParam::size() + 1) / 2;
    using Prototype = xvec::simd::resize_t<lsize, TypeParam>;

    const auto original = to_array(f.v0);

    std::array<typename TypeParam::value_type, lsize> lowerExpected{};
    for (std::size_t i = 0; i < lsize; ++i)
      lowerExpected[i] = original[i];

    std::array<typename TypeParam::value_type, TypeParam::size() - lsize> upperExpected{};
    for (std::size_t i = 0; i < upperExpected.size(); ++i)
      upperExpected[i] = original[lsize + i];

    const auto [l, u] = xvec::simd::chunk<Prototype>(f.v0);
    BOOST_TEST(to_array(l) == lowerExpected, boost::test_tools::per_element());
    BOOST_TEST(to_array(u) == upperExpected, boost::test_tools::per_element());
    BOOST_TEST(to_array(cat(l, u)) == original, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    static_assert([=] {
      constexpr auto v = GetConstexprRandomVector<TypeParam, 22>();
      constexpr auto p = xvec::simd::chunk<Prototype>(v);
      constexpr auto lce = std::get<0>(p);
      constexpr auto uce = std::get<1>(p);

      std::array<typename TypeParam::value_type, lsize> le{};
      for (std::size_t i = 0; i < lsize; ++i) le[i] = v[i];

      std::array<typename TypeParam::value_type, TypeParam::size() - lsize> ue{};
      for (std::size_t i = 0; i < ue.size(); ++i) ue[i] = v[lsize + i];

      return (to_array(lce) == le) &&
             (to_array(uce) == ue) &&
             (to_array(cat(lce, uce)) == to_array(v));
    }());
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SplitIntoMultiplesOf3, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;
  constexpr int pieceSize = 3;

  const auto original = to_array(f.v0);
  const auto pieces = xvec::simd::chunk<3>(f.v0);

  constexpr int numPieces = (TypeParam::size + (pieceSize - 1)) / pieceSize;
  BOOST_REQUIRE_EQUAL(std::tuple_size_v<decltype(pieces)>, numPieces);

  for_n<numPieces>([&](auto i){
    const auto computedPiece = std::get<i>(pieces);

    constexpr int offsetInOriginal = pieceSize * i;
    constexpr int expectedPieceSize = std::min<int>(TypeParam::size - offsetInOriginal, pieceSize);

    std::array<typename TypeParam::value_type, expectedPieceSize> expectedPiece{};
    for (int j = 0; j < expectedPieceSize; ++j)
      expectedPiece[j] = original[offsetInOriginal + j];

    BOOST_TEST(to_array(computedPiece) == expectedPiece, boost::test_tools::per_element());
  });

  if constexpr (TypeParam::size % pieceSize == 0)
  {
    for (std::size_t i = 0; i < pieces.size(); ++i)
    {
      std::array<typename TypeParam::value_type, pieceSize> expectedPiece{};
      for (int j = 0; j < pieceSize; ++j)
        expectedPiece[j] = original[i * pieceSize + j];

      BOOST_TEST(to_array(pieces[i]) == expectedPiece, boost::test_tools::per_element());
    }
  }
}
