//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

using namespace xvec::simd;

template<int _size, xvec::simd::vec_or_mask_type _S>
constexpr auto subrange(int offset, const _S& v)
{
  static_assert(_size <= _S::size);
  std::array<typename _S::value_type, _size> result{};
  for (int i = 0; i < _size; ++i)
    result[i] = v[i + offset];
  return result;
}

/// The tests are only run on the unsigned integers as they are all implemented in
/// terms of other permutes, which are rigorously tested. These tests only check
/// the basic operation works and don't need to be exhaustive.

BOOST_AUTO_TEST_CASE_TEMPLATE(Take, TypeParam, PermuteTestTypes)
{
  auto take_expected = []<int N, typename V>(std::integral_constant<int, N>, const V& v) constexpr {
    std::array<typename V::value_type, N> e{};
    for (int i = 0; i < N; ++i) e[i] = v[i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() >= 2)
    BOOST_TEST(to_array(take<2>(f.v0)) == take_expected(std::integral_constant<int, 2>{}, f.v0),
               boost::test_tools::per_element());

  if constexpr (TypeParam::size() > 7)
    BOOST_TEST(to_array(take<7>(f.v0)) == take_expected(std::integral_constant<int, 7>{}, f.v0),
               boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 101>();
    if constexpr (TypeParam::size() >= 2)
      static_assert(to_array(take<2>(v)) == take_expected(std::integral_constant<int, 2>{}, v));
    if constexpr (TypeParam::size() > 7)
      static_assert(to_array(take<7>(v)) == take_expected(std::integral_constant<int, 7>{}, v));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Drop, TypeParam, PermuteTestTypes)
{
  auto drop_expected = []<int N, typename V>(std::integral_constant<int, N>, const V& v) constexpr {
    std::array<typename V::value_type, V::size - N> e{};
    for (std::size_t i = 0; i < e.size(); ++i) e[i] = v[N + i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() > 2)
    BOOST_TEST(to_array(drop<2>(f.v0)) == drop_expected(std::integral_constant<int, 2>{}, f.v0),
               boost::test_tools::per_element());

  if constexpr (TypeParam::size() > 7)
    BOOST_TEST(to_array(drop<7>(f.v0)) == drop_expected(std::integral_constant<int, 7>{}, f.v0),
               boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 102>();
    if constexpr (TypeParam::size() > 2)
      static_assert(to_array(drop<2>(v)) == drop_expected(std::integral_constant<int, 2>{}, v));
    if constexpr (TypeParam::size() > 7)
      static_assert(to_array(drop<7>(v)) == drop_expected(std::integral_constant<int, 7>{}, v));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GrowValueInit, TypeParam, PermuteTestTypes)
{
  constexpr int N = 5;

  auto grow_default_expected = []<int NewSize, typename V>(std::integral_constant<int, NewSize>, const V& v) constexpr {
    std::array<typename V::value_type, NewSize> e{};
    for (std::size_t i = 0; i < V::size; ++i) e[i] = v[i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;
  constexpr int NewSize = TypeParam::size + N;

  BOOST_TEST(to_array(grow<NewSize>(f.v0)) == grow_default_expected(std::integral_constant<int, NewSize>{}, f.v0),
             boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 103>();
    static_assert(to_array(grow<NewSize>(v)) == grow_default_expected(std::integral_constant<int, NewSize>{}, v));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GrowNewValue, TypeParam, PermuteTestTypes)
{
  constexpr int N = 5;
  constexpr int NewSize = TypeParam::size + N;

  auto grow_fill_expected = []<int Size, typename V>(std::integral_constant<int, Size>, const V& v, typename V::value_type fill) constexpr {
    std::array<typename V::value_type, Size> e{};
    for (auto& x : e) x = fill;
    for (std::size_t i = 0; i < V::size; ++i) e[i] = v[i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;
  auto fillValue = f.v0[0];

  BOOST_TEST(to_array(grow<NewSize>(f.v0, fillValue)) ==
             grow_fill_expected(std::integral_constant<int, NewSize>{}, f.v0, fillValue),
             boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 104>();
    constexpr auto fill = typename TypeParam::value_type(1);
    static_assert(to_array(grow<NewSize>(v, fill)) ==
                  grow_fill_expected(std::integral_constant<int, NewSize>{}, v, fill));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Extract, TypeParam, PermuteTestTypes)
{
  auto extract_expected = []<int Offset, int End, typename V>(std::integral_constant<int, Offset>,
                                                               std::integral_constant<int, End>,
                                                               const V& v) constexpr {
    constexpr int Count = End - Offset;
    std::array<typename V::value_type, Count> e{};
    for (int i = 0; i < Count; ++i) e[i] = v[Offset + i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size() > 4)
  {
    BOOST_TEST(to_array(extract<0, 2>(f.v0)) ==
               extract_expected(std::integral_constant<int, 0>{}, std::integral_constant<int, 2>{}, f.v0),
               boost::test_tools::per_element());

    BOOST_TEST(to_array(extract<1, 4>(f.v0)) ==
               extract_expected(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{}, f.v0),
               boost::test_tools::per_element());
  }

  if constexpr (TypeParam::size() >= 8)
  {
    BOOST_TEST(to_array(extract<3, 8>(f.v0)) ==
               extract_expected(std::integral_constant<int, 3>{}, std::integral_constant<int, 8>{}, f.v0),
               boost::test_tools::per_element());
  }

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 105>();
    if constexpr (TypeParam::size() > 4)
    {
      static_assert(to_array(extract<0, 2>(v)) ==
                    extract_expected(std::integral_constant<int, 0>{}, std::integral_constant<int, 2>{}, v));
      static_assert(to_array(extract<1, 4>(v)) ==
                    extract_expected(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{}, v));
    }
    if constexpr (TypeParam::size() >= 8)
    {
      static_assert(to_array(extract<3, 8>(v)) ==
                    extract_expected(std::integral_constant<int, 3>{}, std::integral_constant<int, 8>{}, v));
    }
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(RepeatAll, TypeParam, PermuteTestTypes)
{
  constexpr int N = 3;

  auto repeat_all_expected = []<int K, typename V>(std::integral_constant<int, K>, const V& v) constexpr {
    std::array<typename V::value_type, V::size * K> e{};
    for (int r = 0; r < K; ++r)
      for (std::size_t i = 0; i < V::size; ++i)
        e[r * V::size + i] = v[i];
    return e;
  };

  if constexpr (TypeParam::size * N < 128)
  {
    PermuteTestFixture<TypeParam> f;
    BOOST_TEST(to_array(repeat_all<N>(f.v0)) ==
               repeat_all_expected(std::integral_constant<int, N>{}, f.v0),
               boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto v = GetConstexprRandomVector<TypeParam, 106>();
      static_assert(to_array(repeat_all<N>(v)) ==
                    repeat_all_expected(std::integral_constant<int, N>{}, v));
    }
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(RepeatEach, TypeParam, PermuteTestTypes)
{
  constexpr int N = 3;

  auto repeat_each_expected = []<int K, typename V>(std::integral_constant<int, K>, const V& v) constexpr {
    std::array<typename V::value_type, V::size * K> e{};
    for (std::size_t i = 0; i < V::size; ++i)
      for (int j = 0; j < K; ++j)
        e[i * K + j] = v[i];
    return e;
  };

  if constexpr (TypeParam::size * N < 128)
  {
    PermuteTestFixture<TypeParam> f;
    BOOST_TEST(to_array(repeat_each<N>(f.v0)) ==
               repeat_each_expected(std::integral_constant<int, N>{}, f.v0),
               boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
    {
      constexpr auto v = GetConstexprRandomVector<TypeParam, 107>();
      static_assert(to_array(repeat_each<N>(v)) ==
                    repeat_each_expected(std::integral_constant<int, N>{}, v));
    }
#endif
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Stride, TypeParam, PermuteTestTypes)
{
  constexpr int N = 5;

  auto stride_expected = []<int K, typename V>(std::integral_constant<int, K>, const V& v) constexpr {
    constexpr int outSize = (V::size + K - 1) / K;
    std::array<typename V::value_type, outSize> e{};
    for (int i = 0; i < outSize; ++i)
      e[i] = v[i * K];
    return e;
  };

  PermuteTestFixture<TypeParam> f;
  BOOST_TEST(to_array(stride<N>(f.v0)) ==
             stride_expected(std::integral_constant<int, N>{}, f.v0),
             boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 108>();
    static_assert(to_array(stride<N>(v)) ==
                  stride_expected(std::integral_constant<int, N>{}, v));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Reverse, TypeParam, PermuteTestTypes)
{
  auto reverse_expected = []<typename V>(const V& v) constexpr {
    std::array<typename V::value_type, V::size> e{};
    for (std::size_t i = 0; i < V::size; ++i)
      e[i] = v[V::size - 1 - i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;
  BOOST_TEST(to_array(reverse(f.v0)) == reverse_expected(f.v0),
             boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 109>();
    static_assert(to_array(reverse(v)) == reverse_expected(v));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Rotate, TypeParam, PermuteTestTypes)
{
  auto rotate_expected = []<int N, typename V>(std::integral_constant<int, N>, const V& v) constexpr {
    std::array<typename V::value_type, V::size> expected{};
    for (std::size_t i = 0; i < V::size; ++i)
      expected[i] = v[(i + N) % V::size];
    return expected;
  };

  PermuteTestFixture<TypeParam> f;

  auto test_rt = [&]<int N>(std::integral_constant<int, N>) {
    BOOST_TEST(to_array(rotate<N>(f.v0)) ==
               rotate_expected(std::integral_constant<int, N>{}, f.v0),
               boost::test_tools::per_element());
  };

  test_rt(std::integral_constant<int, 0>{});
  test_rt(std::integral_constant<int, 1>{});
  test_rt(std::integral_constant<int, TypeParam::size / 2>{});
  test_rt(std::integral_constant<int, TypeParam::size - 1>{});

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 110>();
    static_assert(to_array(rotate<0>(v)) ==
                  rotate_expected(std::integral_constant<int, 0>{}, v));
    static_assert(to_array(rotate<1>(v)) ==
                  rotate_expected(std::integral_constant<int, 1>{}, v));
    static_assert(to_array(rotate<TypeParam::size / 2>(v)) ==
                  rotate_expected(std::integral_constant<int, TypeParam::size / 2>{}, v));
    static_assert(to_array(rotate<TypeParam::size - 1>(v)) ==
                  rotate_expected(std::integral_constant<int, TypeParam::size - 1>{}, v));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Zip, TypeParam, PermuteTestTypes)
{
  auto zip_expected_2 = []<typename A, typename B>(const A& a, const B& b) constexpr {
    constexpr int cols = (A::size < B::size) ? A::size : B::size;
    std::array<typename A::value_type, cols * 2> e{};
    for (int i = 0; i < cols; ++i)
    {
      e[2 * i] = a[i];
      e[2 * i + 1] = b[i];
    }
    return e;
  };

  PermuteTestFixture<TypeParam> f;

  // Zip of one returns the same thing.
  BOOST_TEST(to_array(zip(f.v0)) == to_array(f.v0), boost::test_tools::per_element());

  // Zip with itself is like repeating each element. Only do so if this fits within the allowable simd sizes.
  if constexpr (TypeParam::size * 2 <= xvec::simd::detail::max_fixed_size<typename TypeParam::value_type>)
  {
    BOOST_TEST(to_array(zip(f.v0, f.v0)) == to_array(repeat_each<2>(f.v0)),
               boost::test_tools::per_element());
  }

  // Zip different sizes returns smallest.
  if constexpr (TypeParam::size > 4)
  {
    auto smaller = take<4>(f.v1);
    auto z = zip(f.v0, smaller);
    static_assert(decltype(z)::size == 8, "Two rows of 4, which is the smallest size");

    BOOST_TEST(to_array(z) == zip_expected_2(f.v0, smaller), boost::test_tools::per_element());
  }

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v0 = GetConstexprRandomVector<TypeParam, 201>();
    constexpr auto v1 = GetConstexprRandomVector<TypeParam, 202>();

    static_assert(to_array(zip(v0)) == to_array(v0));

    if constexpr (TypeParam::size * 2 <= xvec::simd::detail::max_fixed_size<typename TypeParam::value_type>)
    {
      static_assert(to_array(zip(v0, v0)) == to_array(repeat_each<2>(v0)));
    }

    if constexpr (TypeParam::size > 4)
    {
      constexpr auto smaller = take<4>(v1);
      constexpr auto z = zip(v0, smaller);
      static_assert(decltype(z)::size == 8);
      static_assert(to_array(z) == zip_expected_2(v0, smaller));
    }
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Chunk, TypeParam, PermuteTestTypes)
{
  constexpr int pieceSize = 3;

  auto chunk_piece_expected = []<int Offset, int Count, typename V>(std::integral_constant<int, Offset>,
                                                                     std::integral_constant<int, Count>,
                                                                     const V& v) constexpr {
    std::array<typename V::value_type, Count> e{};
    for (int i = 0; i < Count; ++i) e[i] = v[Offset + i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;

  // Split into pieces containing no more than pieceSize elements.
  const auto pieces = chunk<pieceSize>(f.v0);

  constexpr int numPieces = (TypeParam::size + (pieceSize - 1)) / pieceSize;
  BOOST_REQUIRE_EQUAL(std::tuple_size_v<decltype(pieces)>, numPieces);

  for_n<numPieces>([&](auto i) {
    auto computedPiece = std::get<i>(pieces);

    constexpr int offsetInOriginal = pieceSize * i;
    constexpr int expectedPieceSize = std::min<int>(TypeParam::size - offsetInOriginal, pieceSize);
    const auto expectedPiece =
      chunk_piece_expected(std::integral_constant<int, offsetInOriginal>{},
                           std::integral_constant<int, expectedPieceSize>{},
                           f.v0);

    BOOST_TEST(to_array(computedPiece) == expectedPiece, boost::test_tools::per_element());
  });

  // If all pieces are equal an array is returned instead of a tuple.
  if constexpr (TypeParam::size % pieceSize == 0)
  {
    for (std::size_t i = 0; i < pieces.size(); ++i)
    {
      const auto expectedPiece = subrange<pieceSize>(static_cast<int>(i) * pieceSize, f.v0);
      BOOST_TEST(to_array(pieces[i]) == expectedPiece, boost::test_tools::per_element());
    }
  }

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 203>();
    constexpr auto cePieces = chunk<pieceSize>(v);
    static_assert(std::tuple_size_v<decltype(cePieces)> == numPieces);

    for_n<numPieces>([&](auto i) {
      constexpr int offsetInOriginal = pieceSize * i;
      constexpr int expectedPieceSize = std::min<int>(TypeParam::size - offsetInOriginal, pieceSize);
      constexpr auto expectedPiece =
        chunk_piece_expected(std::integral_constant<int, offsetInOriginal>{},
                             std::integral_constant<int, expectedPieceSize>{},
                             v);
      static_assert(to_array(std::get<i>(cePieces)) == expectedPiece);
    });
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Unzip, TypeParam, PermuteTestTypes)
{
  auto unzip_expected_3 = []<typename V>(const V& v) constexpr {
    constexpr int n0 = (V::size + 2) / 3;
    constexpr int n1 = (V::size + 1) / 3;
    constexpr int n2 = (V::size + 0) / 3;

    std::array<typename V::value_type, n0> a0{};
    std::array<typename V::value_type, n1> a1{};
    std::array<typename V::value_type, n2> a2{};

    int i0 = 0, i1 = 0, i2 = 0;
    for (int i = 0; i < V::size; ++i)
    {
      if ((i % 3) == 0) a0[i0++] = v[i];
      else if ((i % 3) == 1) a1[i1++] = v[i];
      else a2[i2++] = v[i];
    }

    return std::tuple{a0, a1, a2};
  };

  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size >= 3)
  {
    auto computed = unzip<3>(f.v0);
    const auto expected = unzip_expected_3(f.v0);

    BOOST_TEST(to_array(std::get<0>(computed)) == std::get<0>(expected), boost::test_tools::per_element());
    BOOST_TEST(to_array(std::get<1>(computed)) == std::get<1>(expected), boost::test_tools::per_element());
    BOOST_TEST(to_array(std::get<2>(computed)) == std::get<2>(expected), boost::test_tools::per_element());
  }

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 204>();
    if constexpr (TypeParam::size >= 3)
    {
      constexpr auto computed = unzip<3>(v);
      constexpr auto expected = unzip_expected_3(v);

      static_assert(to_array(std::get<0>(computed)) == std::get<0>(expected));
      static_assert(to_array(std::get<1>(computed)) == std::get<1>(expected));
      static_assert(to_array(std::get<2>(computed)) == std::get<2>(expected));
    }
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Transpose, TypeParam, PermuteTestTypes)
{
  auto transpose_expected = []<int ROWS, int COLS, typename V>(std::integral_constant<int, ROWS>,
                                                                std::integral_constant<int, COLS>,
                                                                const V& v) constexpr {
    std::array<typename V::value_type, V::size> e{};
    for (int idx = 0; idx < V::size; ++idx)
    {
      const int src = (idx / ROWS) + (idx % ROWS) * COLS;
      e[idx] = v[src];
    }
    return e;
  };

  PermuteTestFixture<TypeParam> f;

  // Choosing what size to transpose by is hard for arbitrary sizes so limit it to multiples of 4.
  if constexpr (TypeParam::size % 4 == 0)
  {
    constexpr int numRows = TypeParam::size / 4;

    auto computed = transpose<numRows, 4>(f.v0);
    BOOST_TEST(to_array(computed) ==
               transpose_expected(std::integral_constant<int, numRows>{},
                                  std::integral_constant<int, 4>{},
                                  f.v0),
               boost::test_tools::per_element());
  }

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v = GetConstexprRandomVector<TypeParam, 205>();
    if constexpr (TypeParam::size % 4 == 0)
    {
      constexpr int numRows = TypeParam::size / 4;
      constexpr auto computed = transpose<numRows, 4>(v);
      static_assert(to_array(computed) ==
                    transpose_expected(std::integral_constant<int, numRows>{},
                                       std::integral_constant<int, 4>{},
                                       v));
    }
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SetElement, TypeParam, PermuteTestTypes)
{
  auto set_element_expected = []<typename V>(const V& v, int index, typename V::value_type newValue) constexpr {
    std::array<typename V::value_type, V::size> e{};
    for (int i = 0; i < V::size; ++i)
      e[i] = (i == index) ? newValue : v[i];
    return e;
  };

  PermuteTestFixture<TypeParam> f;

  if constexpr (TypeParam::size > 2)
  {
    int index = 2;
    typename TypeParam::value_type newValue = [=]() {
      if constexpr (xvec::simd::vec_type<TypeParam>) return f.v1[0];
      else return !f.v0[2];
    }();

    auto computed = set_element(f.v0, index, newValue);
    BOOST_TEST(to_array(computed) == set_element_expected(f.v0, index, newValue),
               boost::test_tools::per_element());
  }

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto v0 = GetConstexprRandomVector<TypeParam, 206>();
    constexpr auto v1 = GetConstexprRandomVector<TypeParam, 207>();

    if constexpr (TypeParam::size > 2)
    {
      constexpr int index = 2;
      constexpr typename TypeParam::value_type newValue = [=]() constexpr {
        if constexpr (xvec::simd::vec_type<TypeParam>) return v1[0];
        else return !v0[2];
      }();

      constexpr auto computed = set_element(v0, index, newValue);
      static_assert(to_array(computed) == set_element_expected(v0, index, newValue));
    }
  }
#endif
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprSimplePermute, TypeParam, PermuteTestTypes)
{
  constexpr auto v = GetConstexprRandomVector<TypeParam, 0>();

  auto reverse_expected = []<typename V>(const V& vv) constexpr {
    std::array<typename V::value_type, V::size> e{};
    for (std::size_t i = 0; i < V::size; ++i)
      e[i] = vv[V::size - 1 - i];
    return e;
  };

  auto take_expected = []<int N, typename V>(std::integral_constant<int, N>, const V& vv) constexpr {
    std::array<typename V::value_type, N> e{};
    for (int i = 0; i < N; ++i) e[i] = vv[i];
    return e;
  };

  // Take
  if constexpr (TypeParam::size() > 7)
  {
    constexpr auto take7 = take<7>(v);
    static_assert(to_array(take7) == take_expected(std::integral_constant<int, 7>{}, v));
    BOOST_TEST(to_array(take7) == take_expected(std::integral_constant<int, 7>{}, v),
               boost::test_tools::per_element());
  }

  // Reverse
  constexpr auto rev = reverse(v);
  static_assert(to_array(rev) == reverse_expected(v));
  BOOST_TEST(to_array(rev) == reverse_expected(v), boost::test_tools::per_element());
}
#endif

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprCat, TypeParam, PermuteTestTypes)
{
  constexpr auto v0 = GetConstexprRandomVector<TypeParam, 0>();
  constexpr auto v1 = GetConstexprRandomVector<TypeParam, 1>();

  constexpr auto wrapper0 = to_array(v0);
  constexpr auto wrapper1 = to_array(v1);

  auto cat2_expected = []<typename A, typename B>(const A& a, const B& b) constexpr {
    std::array<typename A::value_type, A::size + B::size> e{};
    for (std::size_t i = 0; i < A::size; ++i) e[i] = a[i];
    for (std::size_t i = 0; i < B::size; ++i) e[A::size + i] = b[i];
    return e;
  };

  auto cat3_expected = []<typename A, typename B, typename C>(const A& a, const B& b, const C& c) constexpr {
    std::array<typename A::value_type, A::size + B::size + C::size> e{};
    for (std::size_t i = 0; i < A::size; ++i) e[i] = a[i];
    for (std::size_t i = 0; i < B::size; ++i) e[A::size + i] = b[i];
    for (std::size_t i = 0; i < C::size; ++i) e[A::size + B::size + i] = c[i];
    return e;
  };

  // Cat of one vec (get a copy back)
  constexpr auto c0 = cat(v0);
  static_assert(to_array(c0) == wrapper0);
  BOOST_TEST(to_array(c0) == wrapper0, boost::test_tools::per_element());

  // Cat of two but only if it won't exceed simd's size limits.
  if constexpr (TypeParam::size * 2 <= xvec::simd::detail::max_fixed_size<typename TypeParam::value_type>)
  {
    constexpr auto c01 = cat(v0, v1);
    constexpr auto expected01 = cat2_expected(v0, v1);
    static_assert(to_array(c01) == expected01);
    BOOST_TEST(to_array(c01) == expected01, boost::test_tools::per_element());

    // Cat of three but only if it won't exceed simd's size limits.
    if constexpr (TypeParam::size * 3 <= xvec::simd::detail::max_fixed_size<typename TypeParam::value_type>)
    {
      constexpr auto c010 = cat(v0, v1, v0);
      constexpr auto expected010 = cat3_expected(v0, v1, v0);
      static_assert(to_array(c010) == expected010);
      BOOST_TEST(to_array(c010) == expected010, boost::test_tools::per_element());
    }
  }

  (void)wrapper1;
}
#endif
