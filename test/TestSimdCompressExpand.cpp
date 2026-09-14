//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(CompressByFullMask, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;
  using _Tp = typename TypeParam::value_type;

  // Don't bother permuting any little simd objects.
  if constexpr (TypeParam::size() >= 4)
  {
    auto maskBits = getRandomBitset<TypeParam::size>();
    using CompressionMask = decltype((+TypeParam()) != (+TypeParam())); // Sneaky trick to get a mask from either a vec or a mask.
    auto selector = CompressionMask(maskBits);

    constexpr auto fillValue = _Tp(1); // Valid on all supported types, but non-zero.
    std::array<_Tp, TypeParam::size> expected = {};
    std::fill_n(expected.begin(), expected.size(), fillValue);
    int nextOutput = 0;
    for (std::size_t i=0; i<expected.size(); ++i)
    {
      if (maskBits[i]) {
        expected[nextOutput] = f.v0[i];
        nextOutput += 1;
      }
    }

    // Undefined values will be present for unused elements, so only check the first values.
    const auto computedWithUndefined = compress(f.v0, selector);
    auto numToCheck = maskBits.count();
    BOOST_TEST(std::views::take(to_array(computedWithUndefined), numToCheck) == std::views::take(expected, numToCheck), boost::test_tools::per_element());

    // Overloaded version allows data to be inserted in the unused elements.
    const auto computedWithFill = compress(f.v0, selector, fillValue);
    BOOST_TEST(to_array(computedWithFill) == expected, boost::test_tools::per_element());

    // Change the selector to be full to copy the entire input mask. Equivalent to an identity.
    const auto computedFull = compress(f.v0, CompressionMask(true));
    BOOST_TEST(to_array(computedFull) == to_array(f.v0), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto ceD = GetConstexprRandomVector<TypeParam, 0>();
      constexpr auto ceM = GetConstexprRandomVector<CompressionMask, 1>();
      constexpr auto ceFill = _Tp(1);

      constexpr auto ceComputed = compress(ceD, ceM);            // constexpr path
      constexpr auto ceFillComputed = compress(ceD, ceM, ceFill); // constexpr path with fill

      constexpr auto ceExpectedDefault = [=] {
        std::array<_Tp, TypeParam::size> out{};
        std::size_t next = 0;
        for (std::size_t i = 0; i < TypeParam::size; ++i)
          if (ceM[i])
            out[next++] = ceD[i];
        return out;
      }();

      constexpr auto ceExpectedFill = [=] {
        std::array<_Tp, TypeParam::size> out{};
        for (auto& x : out) x = ceFill;
        std::size_t next = 0;
        for (std::size_t i = 0; i < TypeParam::size; ++i)
          if (ceM[i])
            out[next++] = ceD[i];
        return out;
      }();

      static_assert(to_array(ceComputed) == ceExpectedDefault);
      static_assert(to_array(ceFillComputed) == ceExpectedFill);

      BOOST_TEST(to_array(ceComputed) == ceExpectedDefault, boost::test_tools::per_element());
      BOOST_TEST(to_array(ceFillComputed) == ceExpectedFill, boost::test_tools::per_element());
    }
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ExpandByMask, TypeParam, PermuteTestTypes)
{
  PermuteTestFixture<TypeParam> f;
  using _Tp = typename TypeParam::value_type;

  auto maskBits = getRandomBitset<TypeParam::size>();
  using CompressionMask = decltype((+TypeParam()) != (+TypeParam())); // Sneaky trick to get a mask from either a vec or a mask.
  auto selector = CompressionMask(maskBits);

  // Copy contiguous elements from v0 into the output positions whose mask bit is set.
  std::array<_Tp, TypeParam::size> expected = {};
  int nextInput = 0;
  for (std::size_t i=0; i<expected.size(); ++i)
  {
    if (maskBits[i]) {
      expected[i] = f.v0[nextInput];
      nextInput += 1;
    }
    else
      expected[i] = f.v1[i];
  }

  const auto computed = expand(f.v0, selector, f.v1);
  BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto ceD0 = GetConstexprRandomVector<TypeParam, 0>();
    constexpr auto ceD1 = GetConstexprRandomVector<TypeParam, 1>();
    constexpr auto ceM = GetConstexprRandomVector<CompressionMask, 2>();

    constexpr auto ceComputed = expand(ceD0, ceM, ceD1);

    constexpr auto ceExpected = [=] {
      std::array<_Tp, TypeParam::size> out{};
      std::size_t next = 0;
      for (std::size_t i = 0; i < TypeParam::size; ++i)
        out[i] = ceM[i] ? ceD0[next++] : ceD1[i];
      return out;
    }();

    static_assert(to_array(ceComputed) == ceExpected);
    BOOST_TEST(to_array(ceComputed) == ceExpected, boost::test_tools::per_element());
  }
#endif
}
