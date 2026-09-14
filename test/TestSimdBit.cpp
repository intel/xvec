//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

template<typename TypeParam>
struct BitFixture
{
  using _Tp = typename TypeParam::value_type;
  static constexpr auto numBits = std::numeric_limits<_Tp>::digits;
  static constexpr auto empty = TypeParam();
  static constexpr auto full = TypeParam(_Tp(~_Tp()));
  static constexpr auto lsb = TypeParam(_Tp(1));
  static constexpr auto msb = TypeParam(_Tp(_Tp(1) << (numBits - 1)));
  static constexpr auto allButLsb = TypeParam(_Tp(~_Tp(1)));
  static constexpr auto allButMsb = TypeParam(_Tp(~(_Tp(1) << (numBits - 1))));
  static constexpr auto bit5 = TypeParam(_Tp(16));
  static constexpr auto allButLowerNibble = TypeParam(_Tp(~_Tp(0xf)));
  static constexpr auto oneNibbleSet = TypeParam(_Tp(15));
  static constexpr auto rnd = GetConstexprRandomVector<TypeParam, 0>();
};

template<class V, class S> concept RotlCallable = requires(V v, S s) { xvec::simd::rotl(v, s); };
template<class V, class S> concept RotrCallable = requires(V v, S s) { xvec::simd::rotr(v, s); };

BOOST_AUTO_TEST_CASE(RotateMismatchedSizesIsIllFormed)
{
  using SmallVec = xvec::simd::vec<unsigned int, 4>;
  using LargeVec = xvec::simd::vec<unsigned int, 8>;

  static_assert(!RotlCallable<SmallVec, LargeVec>, "rotl must not be callable with simd arguments of different sizes");
  static_assert(!RotrCallable<SmallVec, LargeVec>, "rotr must not be callable with simd arguments of different sizes");
}

BOOST_AUTO_TEST_CASE(RotateMismatchedElementSizesIsIllFormed)
{
  using LargeElement = xvec::simd::vec<unsigned int, 8>;
  using SmallElement = xvec::simd::vec<unsigned char, 8>;

  static_assert(!RotlCallable<LargeElement, SmallElement>, "rotl must not be callable with simd arguments with different element sizes");
  static_assert(!RotrCallable<LargeElement, SmallElement>, "rotr must not be callable with simd arguments with different element sizes");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CountLeadingZero, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_countl_zero = [](_Tp x) constexpr {
    return static_cast<std::make_signed_t<_Tp>>(std::countl_zero(x));
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(countl_zero(v)) == applyUnaryToArray(v, expected_countl_zero), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(countl_zero(f.empty)) == applyUnaryToArray(f.empty, expected_countl_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_zero(f.lsb)) == applyUnaryToArray(f.lsb, expected_countl_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_zero(f.bit5)) == applyUnaryToArray(f.bit5, expected_countl_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_zero(f.full)) == applyUnaryToArray(f.full, expected_countl_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_zero(f.allButMsb)) == applyUnaryToArray(f.allButMsb, expected_countl_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_zero(f.msb)) == applyUnaryToArray(f.msb, expected_countl_zero), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1001>();
    static_assert(to_array(countl_zero(cv)) == applyUnaryToArray(cv, expected_countl_zero));

    static_assert(to_array(countl_zero(BitFixture<TypeParam>::empty)) == applyUnaryToArray(BitFixture<TypeParam>::empty, expected_countl_zero));
    static_assert(to_array(countl_zero(BitFixture<TypeParam>::lsb)) == applyUnaryToArray(BitFixture<TypeParam>::lsb, expected_countl_zero));
    static_assert(to_array(countl_zero(BitFixture<TypeParam>::bit5)) == applyUnaryToArray(BitFixture<TypeParam>::bit5, expected_countl_zero));
    static_assert(to_array(countl_zero(BitFixture<TypeParam>::full)) == applyUnaryToArray(BitFixture<TypeParam>::full, expected_countl_zero));
    static_assert(to_array(countl_zero(BitFixture<TypeParam>::allButMsb)) == applyUnaryToArray(BitFixture<TypeParam>::allButMsb, expected_countl_zero));
    static_assert(to_array(countl_zero(BitFixture<TypeParam>::msb)) == applyUnaryToArray(BitFixture<TypeParam>::msb, expected_countl_zero));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CountLeadingOne, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_countl_one = [](_Tp x) constexpr {
    return static_cast<std::make_signed_t<_Tp>>(std::countl_one(x));
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(countl_one(v)) == applyUnaryToArray(v, expected_countl_one), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(countl_one(f.full)) == applyUnaryToArray(f.full, expected_countl_one), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_one(f.allButLsb)) == applyUnaryToArray(f.allButLsb, expected_countl_one), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_one(f.allButLowerNibble)) == applyUnaryToArray(f.allButLowerNibble, expected_countl_one), boost::test_tools::per_element());
  BOOST_TEST(to_array(countl_one(f.empty)) == applyUnaryToArray(f.empty, expected_countl_one), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1002>();
    static_assert(to_array(countl_one(cv)) == applyUnaryToArray(cv, expected_countl_one));

    static_assert(to_array(countl_one(BitFixture<TypeParam>::full)) == applyUnaryToArray(BitFixture<TypeParam>::full, expected_countl_one));
    static_assert(to_array(countl_one(BitFixture<TypeParam>::allButLsb)) == applyUnaryToArray(BitFixture<TypeParam>::allButLsb, expected_countl_one));
    static_assert(to_array(countl_one(BitFixture<TypeParam>::allButLowerNibble)) == applyUnaryToArray(BitFixture<TypeParam>::allButLowerNibble, expected_countl_one));
    static_assert(to_array(countl_one(BitFixture<TypeParam>::empty)) == applyUnaryToArray(BitFixture<TypeParam>::empty, expected_countl_one));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CountTrailingZero, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_countr_zero = [](_Tp x) constexpr {
    return static_cast<std::make_signed_t<_Tp>>(std::countr_zero(x));
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(countr_zero(v)) == applyUnaryToArray(v, expected_countr_zero), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(countr_zero(f.empty)) == applyUnaryToArray(f.empty, expected_countr_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countr_zero(f.lsb)) == applyUnaryToArray(f.lsb, expected_countr_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countr_zero(f.bit5)) == applyUnaryToArray(f.bit5, expected_countr_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countr_zero(f.full)) == applyUnaryToArray(f.full, expected_countr_zero), boost::test_tools::per_element());
  BOOST_TEST(to_array(countr_zero(f.allButLsb)) == applyUnaryToArray(f.allButLsb, expected_countr_zero), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1003>();
    static_assert(to_array(countr_zero(cv)) == applyUnaryToArray(cv, expected_countr_zero));

    static_assert(to_array(countr_zero(BitFixture<TypeParam>::empty)) == applyUnaryToArray(BitFixture<TypeParam>::empty, expected_countr_zero));
    static_assert(to_array(countr_zero(BitFixture<TypeParam>::lsb)) == applyUnaryToArray(BitFixture<TypeParam>::lsb, expected_countr_zero));
    static_assert(to_array(countr_zero(BitFixture<TypeParam>::bit5)) == applyUnaryToArray(BitFixture<TypeParam>::bit5, expected_countr_zero));
    static_assert(to_array(countr_zero(BitFixture<TypeParam>::full)) == applyUnaryToArray(BitFixture<TypeParam>::full, expected_countr_zero));
    static_assert(to_array(countr_zero(BitFixture<TypeParam>::allButLsb)) == applyUnaryToArray(BitFixture<TypeParam>::allButLsb, expected_countr_zero));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CountTrailingOne, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_countr_one = [](_Tp x) constexpr {
    return static_cast<std::make_signed_t<_Tp>>(std::countr_one(x));
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(countr_one(v)) == applyUnaryToArray(v, expected_countr_one), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(countr_one(f.full)) == applyUnaryToArray(f.full, expected_countr_one), boost::test_tools::per_element());
  BOOST_TEST(to_array(countr_one(f.allButLsb)) == applyUnaryToArray(f.allButLsb, expected_countr_one), boost::test_tools::per_element());

  const auto lowNibbleOnes = ~TypeParam(_Tp(0xf0));
  BOOST_TEST(to_array(countr_one(lowNibbleOnes)) == applyUnaryToArray(lowNibbleOnes, expected_countr_one), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1004>();
    static_assert(to_array(countr_one(cv)) == applyUnaryToArray(cv, expected_countr_one));

    static_assert(to_array(countr_one(BitFixture<TypeParam>::full)) == applyUnaryToArray(BitFixture<TypeParam>::full, expected_countr_one));
    static_assert(to_array(countr_one(BitFixture<TypeParam>::allButLsb)) == applyUnaryToArray(BitFixture<TypeParam>::allButLsb, expected_countr_one));

    constexpr auto lowNibbleOnesCe = ~TypeParam(_Tp(0xf0));
    static_assert(to_array(countr_one(lowNibbleOnesCe)) == applyUnaryToArray(lowNibbleOnesCe, expected_countr_one));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(PopulationCount, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_popcount = [](_Tp x) constexpr {
    return static_cast<std::make_signed_t<_Tp>>(std::popcount(x));
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(popcount(v)) == applyUnaryToArray(v, expected_popcount), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(popcount(f.empty)) == applyUnaryToArray(f.empty, expected_popcount), boost::test_tools::per_element());
  BOOST_TEST(to_array(popcount(f.lsb)) == applyUnaryToArray(f.lsb, expected_popcount), boost::test_tools::per_element());
  BOOST_TEST(to_array(popcount(f.oneNibbleSet)) == applyUnaryToArray(f.oneNibbleSet, expected_popcount), boost::test_tools::per_element());
  BOOST_TEST(to_array(popcount(f.full)) == applyUnaryToArray(f.full, expected_popcount), boost::test_tools::per_element());
  BOOST_TEST(to_array(popcount(f.allButMsb)) == applyUnaryToArray(f.allButMsb, expected_popcount), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1005>();
    static_assert(to_array(popcount(cv)) == applyUnaryToArray(cv, expected_popcount));

    static_assert(to_array(popcount(BitFixture<TypeParam>::empty)) == applyUnaryToArray(BitFixture<TypeParam>::empty, expected_popcount));
    static_assert(to_array(popcount(BitFixture<TypeParam>::lsb)) == applyUnaryToArray(BitFixture<TypeParam>::lsb, expected_popcount));
    static_assert(to_array(popcount(BitFixture<TypeParam>::oneNibbleSet)) == applyUnaryToArray(BitFixture<TypeParam>::oneNibbleSet, expected_popcount));
    static_assert(to_array(popcount(BitFixture<TypeParam>::full)) == applyUnaryToArray(BitFixture<TypeParam>::full, expected_popcount));
    static_assert(to_array(popcount(BitFixture<TypeParam>::allButMsb)) == applyUnaryToArray(BitFixture<TypeParam>::allButMsb, expected_popcount));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BitWidth, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_bit_width = [](_Tp x) constexpr {
    return static_cast<std::make_signed_t<_Tp>>(std::bit_width(x));
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(bit_width(v)) == applyUnaryToArray(v, expected_bit_width), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1006>();
    static_assert(to_array(bit_width(cv)) == applyUnaryToArray(cv, expected_bit_width));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BitFloor, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_bit_floor = [](_Tp x) constexpr -> _Tp { return static_cast<_Tp>(std::bit_floor(x)); };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(bit_floor(v)) == applyUnaryToArray(v, expected_bit_floor), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(bit_floor(f.empty)) == applyUnaryToArray(f.empty, expected_bit_floor), boost::test_tools::per_element());
  BOOST_TEST(to_array(bit_floor(f.full)) == applyUnaryToArray(f.full, expected_bit_floor), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1007>();
    static_assert(to_array(bit_floor(cv)) == applyUnaryToArray(cv, expected_bit_floor));

    static_assert(to_array(bit_floor(BitFixture<TypeParam>::empty)) == applyUnaryToArray(BitFixture<TypeParam>::empty, expected_bit_floor));
    static_assert(to_array(bit_floor(BitFixture<TypeParam>::full)) == applyUnaryToArray(BitFixture<TypeParam>::full, expected_bit_floor));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BitCeil, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_bit_ceil = [](_Tp x) constexpr -> _Tp { return static_cast<_Tp>(std::bit_ceil(x)); };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max() / 2);
  BOOST_TEST(to_array(bit_ceil(v)) == applyUnaryToArray(v, expected_bit_ceil), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(bit_ceil(f.empty)) == applyUnaryToArray(f.empty, expected_bit_ceil), boost::test_tools::per_element());
  BOOST_TEST(to_array(bit_ceil(f.lsb)) == applyUnaryToArray(f.lsb, expected_bit_ceil), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1008>() / uint8_t(2);
    static_assert(to_array(bit_ceil(cv)) == applyUnaryToArray(cv, expected_bit_ceil));

    static_assert(to_array(bit_ceil(BitFixture<TypeParam>::empty)) == applyUnaryToArray(BitFixture<TypeParam>::empty, expected_bit_ceil));
    static_assert(to_array(bit_ceil(BitFixture<TypeParam>::lsb)) == applyUnaryToArray(BitFixture<TypeParam>::lsb, expected_bit_ceil));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(HasSingleBit, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_has_single_bit = [](_Tp x) constexpr { return std::has_single_bit(x); };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(has_single_bit(v)) == applyUnaryToArray(v, expected_has_single_bit), boost::test_tools::per_element());

  BitFixture<TypeParam> f;
  BOOST_TEST(to_array(has_single_bit(f.empty)) == applyUnaryToArray(f.empty, expected_has_single_bit), boost::test_tools::per_element());
  BOOST_TEST(to_array(has_single_bit(f.full)) == applyUnaryToArray(f.full, expected_has_single_bit), boost::test_tools::per_element());
  BOOST_TEST(to_array(has_single_bit(f.lsb)) == applyUnaryToArray(f.lsb, expected_has_single_bit), boost::test_tools::per_element());
  BOOST_TEST(to_array(has_single_bit(f.oneNibbleSet)) == applyUnaryToArray(f.oneNibbleSet, expected_has_single_bit), boost::test_tools::per_element());
  BOOST_TEST(to_array(has_single_bit(f.msb)) == applyUnaryToArray(f.msb, expected_has_single_bit), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1009>();
    static_assert(to_array(has_single_bit(cv)) == applyUnaryToArray(cv, expected_has_single_bit));

    static_assert(to_array(has_single_bit(BitFixture<TypeParam>::empty)) == applyUnaryToArray(BitFixture<TypeParam>::empty, expected_has_single_bit));
    static_assert(to_array(has_single_bit(BitFixture<TypeParam>::full)) == applyUnaryToArray(BitFixture<TypeParam>::full, expected_has_single_bit));
    static_assert(to_array(has_single_bit(BitFixture<TypeParam>::lsb)) == applyUnaryToArray(BitFixture<TypeParam>::lsb, expected_has_single_bit));
    static_assert(to_array(has_single_bit(BitFixture<TypeParam>::oneNibbleSet)) == applyUnaryToArray(BitFixture<TypeParam>::oneNibbleSet, expected_has_single_bit));
    static_assert(to_array(has_single_bit(BitFixture<TypeParam>::msb)) == applyUnaryToArray(BitFixture<TypeParam>::msb, expected_has_single_bit));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ByteSwap, TypeParam, IntegerSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_byteswap = [](_Tp x) constexpr -> _Tp {
    auto bytes = std::bit_cast<std::array<std::byte, sizeof(_Tp)>>(x);
    std::ranges::reverse(bytes);
    return std::bit_cast<_Tp>(bytes);
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(byteswap(v)) == applyUnaryToArray(v, expected_byteswap), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1010>();
    static_assert(to_array(byteswap(cv)) == applyUnaryToArray(cv, expected_byteswap));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BitReverse, TypeParam, UnsignedSimdTypes)
{
  using _Tp = typename TypeParam::value_type;

  auto expected_bit_reverse = [](_Tp x) constexpr -> _Tp {
    _Tp r = 0;
    constexpr int n = std::numeric_limits<_Tp>::digits;
    for (int i = 0; i < n; ++i)
    {
      r = static_cast<_Tp>(r << 1);
      r = static_cast<_Tp>(r | (x & _Tp(1)));
      x = static_cast<_Tp>(x >> 1);
    }
    return r;
  };

  const auto v = GetRandomVector<TypeParam>(std::numeric_limits<_Tp>::max());
  BOOST_TEST(to_array(bit_reverse(v)) == applyUnaryToArray(v, expected_bit_reverse), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto cv = GetConstexprRandomVector<TypeParam, 1011>();
    static_assert(to_array(bit_reverse(cv)) == applyUnaryToArray(cv, expected_bit_reverse));
  }
#endif
}

template<typename T>
class SimdRotateFixture
{
public:
  static constexpr auto numBits = sizeof(typename T::value_type) * 8;

  const T rotates = GetRandomVector<T>(numBits * 10);

  using _UV = xvec::simd::vec<std::make_unsigned_t<typename T::value_type>, T::size()>;
  const _UV v0 = GetRandomVector<_UV>(32);

  const typename T::value_type scalarRotate = getMedian(rotates);
};

BOOST_AUTO_TEST_CASE_TEMPLATE(RotateLeft, TypeParam, IntegerSimdTypes)
{
  SimdRotateFixture<TypeParam> f;
  using _UV = typename SimdRotateFixture<TypeParam>::_UV;

  auto expected_rotl_simd = [](typename _UV::value_type v, typename TypeParam::value_type s) constexpr {
    return std::rotl(v, s);
  };

  auto expected_rotl_scalar = [](typename _UV::value_type v, int s) constexpr {
    return std::rotl(v, s);
  };

  BOOST_TEST(to_array(rotl(f.v0, f.rotates)) == applyBinaryToArray(f.v0, f.rotates, expected_rotl_simd), boost::test_tools::per_element());

  BOOST_TEST(to_array(rotl(f.v0, f.scalarRotate)) == applyUnaryToArray(f.v0, [=](auto v) constexpr { return expected_rotl_scalar(v, f.scalarRotate); }), boost::test_tools::per_element());
  BOOST_TEST(to_array(rotl(f.v0, 1)) == applyUnaryToArray(f.v0, [=](auto v) constexpr { return expected_rotl_scalar(v, 1); }), boost::test_tools::per_element());
  BOOST_TEST(to_array(rotl(f.v0, -1)) == applyUnaryToArray(f.v0, [=](auto v) constexpr { return expected_rotl_scalar(v, -1); }), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto rndRotates = GetConstexprRandomVector<TypeParam, 1012>();
    constexpr auto rndVals = GetConstexprRandomVector<_UV, 1013>();

    static_assert(to_array(rotl(rndVals, rndRotates)) == applyBinaryToArray(rndVals, rndRotates, expected_rotl_simd));
    static_assert(to_array(rotl(rndVals, rndRotates[0])) == applyBinaryToArray(rndVals, TypeParam(rndRotates[0]), expected_rotl_simd));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(RotateRight, TypeParam, UnsignedSimdTypes)
{
  SimdRotateFixture<TypeParam> f;
  using _UV = typename SimdRotateFixture<TypeParam>::_UV;

  auto expected_rotr_simd = [](typename _UV::value_type v, typename TypeParam::value_type s) constexpr {
    return std::rotr(v, s);
  };

  auto expected_rotr_scalar = [](typename _UV::value_type v, int s) constexpr {
    return std::rotr(v, s);
  };

  BOOST_TEST(to_array(rotr(f.v0, f.rotates)) == applyBinaryToArray(f.v0, f.rotates, expected_rotr_simd), boost::test_tools::per_element());

  BOOST_TEST(to_array(rotr(f.v0, f.scalarRotate)) == applyUnaryToArray(f.v0, [=](auto v) constexpr { return expected_rotr_scalar(v, f.scalarRotate); }), boost::test_tools::per_element());
  BOOST_TEST(to_array(rotr(f.v0, 1)) == applyUnaryToArray(f.v0, [=](auto v) constexpr { return expected_rotr_scalar(v, 1); }), boost::test_tools::per_element());
  BOOST_TEST(to_array(rotr(f.v0, -1)) == applyUnaryToArray(f.v0, [=](auto v) constexpr { return expected_rotr_scalar(v, -1); }), boost::test_tools::per_element());

#if defined(_XVEC_HAS_CONSTEXPR)
  {
    constexpr auto rndRotates = GetConstexprRandomVector<TypeParam, 1014>();
    constexpr auto rndVals = GetConstexprRandomVector<_UV, 1015>();

    static_assert(to_array(rotr(rndVals, rndRotates)) == applyBinaryToArray(rndVals, rndRotates, expected_rotr_simd));
    static_assert(to_array(rotr(rndVals, rndRotates[0])) == applyBinaryToArray(rndVals, TypeParam(rndRotates[0]), expected_rotr_simd));
  }
#endif
}
