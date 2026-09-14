//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <functional>

#include <xvec/simd>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(CoreVectorOperations) {
  using Vec = xvec::simd::vec<int, 64>;

  const Vec ascending = xvec::simd::iota<Vec>;
  const Vec result = (ascending + Vec(2)) * Vec(2);

  for (std::size_t i = 0; i < Vec::size(); ++i)
    BOOST_TEST(result[i] == 2 * (static_cast<int>(i) + 2));

  const auto upper_half = ascending > Vec(31);
  BOOST_TEST(xvec::simd::any_of(upper_half));
  BOOST_TEST(!xvec::simd::all_of(upper_half));
  BOOST_TEST(xvec::simd::reduce_count(upper_half) == 32);

  const Vec selected = xvec::simd::select(upper_half, ascending, Vec(0));
  BOOST_TEST(selected[31] == 0);
  BOOST_TEST(selected[32] == 32);
}

BOOST_AUTO_TEST_CASE(IntegerBitwiseOperations) {
  using Vec = xvec::simd::vec<unsigned, 64>;

  const Vec lhs(0xf0u);
  const Vec rhs(0x0fu);

  BOOST_TEST(xvec::simd::all_of((lhs & rhs) == Vec(0u)));
  BOOST_TEST(xvec::simd::all_of((lhs | rhs) == Vec(0xffu)));
  BOOST_TEST(xvec::simd::all_of((rhs << 4) == lhs));
}

BOOST_AUTO_TEST_CASE(ComparisonAndMaskOperations) {
  using Vec = xvec::simd::vec<int, 64>;

  const Vec ascending = xvec::simd::iota<Vec>;
  const auto middle = (ascending > Vec(2)) & (ascending < Vec(6));

  BOOST_TEST(xvec::simd::all_of(ascending < Vec(64)));
  BOOST_TEST(xvec::simd::none_of(ascending < Vec(0)));
  BOOST_TEST(xvec::simd::none_of(ascending > Vec(63)));
  BOOST_TEST(xvec::simd::reduce_count(middle) == 3);
}

BOOST_AUTO_TEST_CASE(MinMaxOperations) {
  using Vec = xvec::simd::vec<int, 64>;

  const Vec ascending = xvec::simd::iota<Vec>;
  const Vec descending = Vec(63) - ascending;
  const Vec minima = xvec::simd::min(ascending, descending);
  const Vec maxima = xvec::simd::max(ascending, descending);

  BOOST_TEST(minima[0] == 0);
  BOOST_TEST(minima[31] == 31);
  BOOST_TEST(maxima[32] == 32);
  BOOST_TEST(maxima[63] == 63);
}

BOOST_AUTO_TEST_CASE(UnaryOperations) {
  using Vec = xvec::simd::vec<int, 64>;

  const Vec values = xvec::simd::iota<Vec> - Vec(32);

  BOOST_TEST(xvec::simd::all_of(-values == Vec(32) - xvec::simd::iota<Vec>));
  BOOST_TEST(xvec::simd::all_of(+values == values));
}

BOOST_AUTO_TEST_CASE(LoadStoreOperations) {
  using Vec = xvec::simd::vec<int, 64>;

  const std::array input{1, 2, 3, 4, 5, 6};
  const Vec loaded = xvec::simd::partial_load<Vec>(input);
  std::array<int, Vec::size()> output{};
  xvec::simd::partial_store(loaded, output);

  for (std::size_t i = 0; i < input.size(); ++i)
    BOOST_TEST(output[i] == input[i]);
  BOOST_TEST(output[6] == 0);
  BOOST_TEST(output[63] == 0);
}

BOOST_AUTO_TEST_CASE(ReductionAndPermutation) {
  using Vec = xvec::simd::vec<int, 64>;

  const Vec ascending = xvec::simd::iota<Vec>;
  const Vec reversed = xvec::simd::reverse(ascending);

  BOOST_TEST(xvec::simd::reduce(ascending) == 2016);
  BOOST_TEST(xvec::simd::reduce_min(ascending) == 0);
  BOOST_TEST(xvec::simd::reduce_max(ascending) == 63);
  BOOST_TEST(reversed[0] == 63);
  BOOST_TEST(reversed[63] == 0);
}

BOOST_AUTO_TEST_CASE(FloatingPointMath) {
  using Vec = xvec::simd::vec<float, 64>;

  const Vec values = xvec::simd::iota<Vec> + Vec(1.0f);
  const Vec squares = values * values;
  const Vec roots = xvec::simd::sqrt(squares);
  const Vec fused = xvec::simd::fma(Vec(2.0f), Vec(3.0f), Vec(4.0f));

  for (std::size_t i = 0; i < Vec::size(); ++i)
    BOOST_TEST(roots[i] == static_cast<float>(i) + 1.0f);
  BOOST_TEST(xvec::simd::all_of(fused == Vec(10.0f)));
}
