//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

// Nasty overloads needed for some FP16 functions since the default libraries don't provide them yet.
#if defined(__FLT16_MIN__)
_Float16 copysign(_Float16 mag, _Float16 sign) { return _Float16(copysign(float(mag), float(sign))); }
_Float16 abs(_Float16 x) { return _Float16(abs(float(x))); }
_Float16 abs(std::complex<_Float16> x) { return _Float16(sqrt(x.real() * x.real() + x.imag() * x.imag())); };
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(Copysign, TypeParam, FloatSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  { // Non-constexpr
    const auto expected = applyBinary (f.v0, f.v1, [](auto l, auto r) { using namespace std; return copysign(l, r); });
    const auto computed = copysign(f.v0, f.v1);
    BOOST_SIMD_EQUAL(computed, expected);
  }

  // TODO: :COMPILER: Make all builtins constexpr - abs isn't at the moment.
  // Constexpr
  // {
  //   constexpr auto vMag = TypeParam::iota();
  //   constexpr auto vSign = TypeParam::iota() - TypeParam::size() / 2;

  //   constexpr auto computed = copysign(vMag, vSign);
  //   const auto expected = applyBinary (vMag, vSign, [](auto l, auto r) { return copysign(l, r); });

  //   BOOST_SIMD_EQUAL(computed, expected);
  // }
}

// Eventually complex abs will be implemented and then the is_complex_v part can be removed.
template<typename _T> using IsAbsable =
  boost::mp11::mp_bool<(requires (_T v) { abs(v); })>;
using AbsableSimdTypes = boost::mp11::mp_copy_if<AllSimdTypes, IsAbsable>;

BOOST_AUTO_TEST_CASE_TEMPLATE(Abs, TypeParam, AbsableSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  using _Tp = typename TypeParam::value_type;

  { // Non-constexpr
    const auto expected = applyUnary (f.v0, [](auto v) {
      if constexpr ((std::integral<_Tp> || std::is_enum_v<_Tp>) && sizeof(_Tp) < sizeof(int))
        return _Tp(abs(v)); // Normal abs promotes to int, but xvec retains type.
      else
        return abs(v);
    });
    const auto computed = abs(f.v0);
    BOOST_SIMD_EQUAL(computed, expected);
  }

  {
    // constexpr auto t0 = TypeParam::iota() - TypeParam::size() / 2; // Create negatives in the first part
    // const auto expected = applyUnary (f.v0, [](auto v) -> decltype(v) { return abs(v); });
    // constexpr auto computed = abs(t0);
    // BOOST_SIMD_EQUAL(computed, expected);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Rcp, TypeParam, FloatSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  {
    // Non-constexpr. Note that rcp is invoked on every vec element using a
    // scalar single-element vec, so this is essentially checking that whatever is applied to
    // a single element gets applied to all elements, which isn't independently
    // testing rcp against another reference implementation. However, it is very
    // difficult to test rcp since it is an approximation which varies from one
    // target to another.
    const auto posV0 = abs(f.v0);
    const auto expected = applyUnary(posV0, [](auto x) {
      xvec::simd::vec<typename TypeParam::value_type, 1> s(x);
      return rcp(s)[0];
    });
    BOOST_SIMD_EQUAL(rcp(posV0), expected);
  }

  {
    // constexpr auto t0 = TypeParam::iota() - TypeParam::size() / 2; // Create negatives in the first part
    // const auto expected = applyUnary (f.v0, [](auto v) -> decltype(v) { return rcp(v); });
    // constexpr auto computed = xvec::rcp(t0);
    // BOOST_SIMD_EQUAL(computed, expected);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Sqrt, TypeParam, FloatSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  {
    // Non-constexpr.
    const auto posV0 = abs(f.v0);
    const auto expected = applyUnary(posV0, [](auto x) {
      const xvec::simd::vec<typename TypeParam::value_type, 1> s(x);
      const auto r = sqrt(s);
      return r[0];
    });
    BOOST_SIMD_EQUAL(sqrt(posV0), expected);
  }

  {
    // constexpr auto t0 = TypeParam::iota() - TypeParam::size() / 2; // Create negatives in the first part
    // const auto expected = applyUnary (f.v0, [](auto v) -> decltype(v) { return xvec::sqrt(v); });
    // constexpr auto computed = xvec::sqrt(t0);
    // BOOST_SIMD_EQUAL(computed, expected);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Rsqrt, TypeParam, FloatSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  {
    // Non-constexpr. Note that rsqrt is invoked on every vec element using a
    // scalar single-element vec, so this is essentially checking that whatever is applied to
    // a single element gets applied to all elements, which isn't independently
    // testing rsqrt against another reference implementation. However, it is very
    // difficult to test rsqrt since it is an approximation which varies from one
    // target to another.
    const auto posV0 = abs(f.v0);
    const auto expected = applyUnary(posV0, [](auto x) {
      const xvec::simd::vec<typename TypeParam::value_type, 1> s(x);
      const auto r = rsqrt(s);
      return r[0];
    });
    BOOST_SIMD_EQUAL(rsqrt(posV0), expected);
  }

  {
    // constexpr auto t0 = TypeParam::iota() - TypeParam::size() / 2; // Create negatives in the first part
    // const auto expected = applyUnary (f.v0, [](auto v) -> decltype(v) { return rsqrt(v); });
    // constexpr auto computed = xvec::rsqrt(t0);
    // BOOST_SIMD_EQUAL(computed, expected);
  }
}
