//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimdTestUtilities.hpp"
#include "TypesToTest.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(ZeroInitialisingConstructor, TypeParam, ComplexSimdTypes)
{
  typename TypeParam::value_type defaultComplexValue;
  std::array<typename TypeParam::value_type, TypeParam::size()> expected; expected.fill(defaultComplexValue);

  TypeParam zeroComplex;
  BOOST_TEST(to_array(zeroComplex) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromComplexPair, TypeParam, ComplexSimdTypes)
{
  constexpr typename TypeParam::value_type ce = {13, -56};
  std::array<typename TypeParam::value_type, TypeParam::size()> expected; expected.fill(ce);

  TypeParam computedFromComplex(ce);
  BOOST_TEST(to_array(computedFromComplex) == expected, boost::test_tools::per_element());

  constexpr TypeParam constExprComplex(ce);
  BOOST_TEST(to_array(constExprComplex) == expected, boost::test_tools::per_element());

  TypeParam directListInitialized({13, -56});
  BOOST_TEST(to_array(directListInitialized) == expected, boost::test_tools::per_element());

  TypeParam copyListInitialized = {{13, -56}};
  BOOST_TEST(to_array(copyListInitialized) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromRealScalar, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  {
    const auto realC = f.getRandomValue().real();
    TypeParam computedFromReal(realC);
    std::array<typename TypeParam::value_type, TypeParam::size()> expected; expected.fill(realC);
    BOOST_TEST(to_array(computedFromReal) == expected, boost::test_tools::per_element());
  }

  {
    constexpr TypeParam constExprFromReal(typename TypeParam::value_type(123));
    std::array<typename TypeParam::value_type, TypeParam::size()> expected; expected.fill(123);
    BOOST_TEST(to_array(constExprFromReal) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromRealSimd, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  using RealVecType = decltype(TypeParam().real());
  const auto realSimd = GetRandomVector<RealVecType>(f.random_limit);

  TypeParam computedFromRealSimd(realSimd);
  const auto expected = applyUnary(realSimd,  [](auto v) -> std::complex<decltype(v)> { return v; });
  BOOST_SIMD_EQUAL(computedFromRealSimd, expected);

  // Build from iota, since that is constexpr.
#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr auto rsc = xvec::simd::iota<RealVecType>;
  constexpr auto constexprFromReal = TypeParam(rsc);
  BOOST_SIMD_EQUAL(constexprFromReal, TypeParam(rsc));
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructFromRealImagSimd, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  using RealVecType = decltype(TypeParam().real());
  const auto realSimd = GetRandomVector<RealVecType>(f.random_limit);
  const auto imagSimd = GetRandomVector<RealVecType>(f.random_limit);

  TypeParam computedFromRealSimd(realSimd, imagSimd);
  const auto expected = applyBinary(realSimd, imagSimd, [](auto sr, auto si) { return std::complex<decltype(sr)>(sr, si); });
  BOOST_SIMD_EQUAL(computedFromRealSimd, expected);

  // Build from iota, since that is constexpr.
#if defined(_XVEC_HAS_CONSTEXPR)
  constexpr auto sc = xvec::simd::iota<RealVecType>;
  constexpr auto constexprFromReal = TypeParam(sc, sc);
  BOOST_SIMD_EQUAL(constexprFromReal, TypeParam(sc, sc));
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructFromOtherSimdType, TypeParam, ComplexSimdTypes)
{
  constexpr auto t = xvec::simd::iota<xvec::simd::vec<std::complex<float>, TypeParam::size()>>;

  typename SimdTestFixture<TypeParam>::test_array_type expected = {};
  std::iota(expected.begin(), expected.end(), 0);

  // constexpr
  #if (_XVEC_HAS_CONSTEXPR)
    constexpr auto computedConstexpr = TypeParam(t);
    BOOST_TEST(to_array(computedConstexpr) == expected, boost::test_tools::per_element());
  #endif

  auto computedDynamic = TypeParam(std::as_const(t));
  BOOST_TEST(to_array(computedDynamic) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GetUnderlyingElements, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  using elementType = typename TypeParam::value_type::value_type;

  // Compute the original elements.
  std::array<elementType, TypeParam::size * 2> originalAsElements;
  for (int i=0; i<TypeParam::size; ++i)
  {
    originalAsElements[i * 2 + 0] = f.v0[i].real();
    originalAsElements[i * 2 + 1] = f.v0[i].imag();
  }
  BOOST_TEST(to_array(simd_bit_cast<elementType>(f.v0)) == originalAsElements, boost::test_tools::per_element());

  // constexpr version
  std::array<elementType, TypeParam::size() * 2> expectedConstexpr = {};
  for (std::size_t i=0; i<expectedConstexpr.size(); ++i)
    expectedConstexpr[i] = i % 2 == 0 ? i / 2 : 0;
  constexpr auto computedConstexpr = xvec::simd::iota<TypeParam>;
  BOOST_TEST(to_array(simd_bit_cast<elementType>(computedConstexpr)) == expectedConstexpr, boost::test_tools::per_element());
}

// There is also a general equality test, but this particular test is more rigorous, as it checks
// that the different combinations of real/imag matches and mismatches get handled properly, which
// the other test is less likely to pick up.
BOOST_AUTO_TEST_CASE_TEMPLATE(ComplexEquality, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // The base simd checks for equality, but this goes a bit further by ensuring that even when the
  // reals or imaginaries of the compared types are the same, an equality is only presented when
  // both are equal.

  // Create a copy of v0 in which all mod0 elements are identical, mod1 elements have mismatched
  // real and mod2 have mismatched imaginary. This means that only every 3rd element is equal.
  alignas(64) typename SimdTestFixture<TypeParam>::test_array_type newV0;
  const auto original = to_array(f.v0);
  std::copy(original.begin(), original.end(), newV0.begin());
  for (std::size_t i=0; i<newV0.size(); ++i)
  {
    if (i % 3 == 1) newV0[i].real(newV0[i].real() + 3);
    else if (i % 3 == 2)  newV0[i].imag(newV0[i].imag() + 3);
  }

  // Create a bitset where every 3rd element is true.
  std::bitset<TypeParam::size()> expectedMask;
  for (std::size_t i=0; i<TypeParam::size(); i+=3)
    expectedMask.set(i);

  BOOST_TEST((f.v0 == *(TypeParam*)newV0.data()).to_bitset() == expectedMask);
  BOOST_TEST((f.v0 != *(TypeParam*)newV0.data()).to_bitset() == ~expectedMask);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Conjugate, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  const auto expected = applyUnary(f.v0,  [](auto v) { return std::conj(v); });
  BOOST_SIMD_EQUAL(xvec::conj(f.v0), expected);

#if defined (_XVEC_HAS_CONSTEXPR)
  constexpr auto ceCmplx = TypeParam([](auto i) { return typename TypeParam::value_type(0, i); });
  BOOST_SIMD_EQUAL(at_compile_time(conj(ceCmplx)), conj(ceCmplx));
#endif
}

/// Test the various scalar operations. These could transform into a constructor call which
/// takes a single half value, and puts it in the real position, but it would be cleaner to do a
/// masked operation on the broadcast scalar.
BOOST_AUTO_TEST_CASE_TEMPLATE(BinaryMultiplicationByRealScalar, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // complex * scalar
  const typename TypeParam::value_type::value_type realScalar = 3;
  const auto expected = applyUnary (f.v0, [=](auto v) { return v * realScalar; });

  // Binary operator.
  BOOST_SIMD_EQUAL(f.v0 * realScalar, expected);
  BOOST_SIMD_EQUAL(realScalar * f.v0, expected);

  // Assignment operator.
  auto nonConst = f.v0;
  nonConst *= realScalar;
  BOOST_SIMD_EQUAL(nonConst, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BinaryAdditionOfRealScalar, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // complex + scalar
  const typename TypeParam::value_type::value_type realScalar = 3;
  const auto expected = applyUnary (f.v0, [=](auto v) { return v + realScalar; });

  // Binary operator.
  BOOST_SIMD_EQUAL(f.v0 + realScalar, expected);
  BOOST_SIMD_EQUAL(realScalar + f.v0, expected);

  // Assignment operator.
  auto nonConst = f.v0;
  nonConst += realScalar;
  BOOST_SIMD_EQUAL(nonConst, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BinarySubtractionOfRealScalar, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // complex - scalar
  const typename TypeParam::value_type::value_type realScalar = 3;
  const auto expected = applyUnary (f.v0, [=](auto v) { return v - realScalar; });

  // Binary operator.
  BOOST_SIMD_EQUAL(f.v0 - realScalar, expected);

  // Assignment operator.
  auto nonConst = f.v0;
  nonConst -= realScalar;
  BOOST_SIMD_EQUAL(nonConst, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(BinarySubtractionFromRealScalar, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // scalar - complex
  const typename TypeParam::value_type::value_type realScalar = 3;
  const auto expected = applyUnary (f.v0, [=](auto v) { return realScalar - v; });

  // Binary operator.
  BOOST_SIMD_EQUAL(realScalar - f.v0, expected);
}

// BOOST_AUTO_TEST_CASE_TEMPLATE(BinaryDivisionByRealScalar, TypeParam, ComplexSimdTypes)
// {
//   SimdTestFixture<TypeParam> f;

//   // complex / realScalar
//   const typename TypeParam::value_type::value_type realScalar = 3;
//   const auto expected = round(applyUnary (f.v0, [=](auto v) { return v / realScalar; }));

//   // Binary operator.
//   BOOST_SIMD_EQUAL(round(f.v0 / realScalar), expected);

//   // Assignment operator.
//   auto nonConst = f.v0;
//   nonConst /= realScalar;
//   BOOST_SIMD_EQUAL(round(nonConst), expected);
// }

BOOST_AUTO_TEST_CASE_TEMPLATE(Fma, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  const auto expected = applyTernary(f.v0, f.v1, f.v2, [](auto v0, auto v1, auto v2) {
    return v0 * v1 + v2; });
  BOOST_SIMD_EQUAL(xvec::fma(f.v0, f.v1, f.v2), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Norm, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  const auto expected = applyUnary(f.v0, [](auto x) { return std::norm(x); });
  BOOST_SIMD_EQUAL(xvec::norm(f.v0), expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ExtractReal, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  const auto expected = applyUnary(f.v0, [](auto x) { return x.real(); });
  BOOST_SIMD_EQUAL(f.v0.real(), expected);
  BOOST_SIMD_EQUAL(xvec::real(f.v0), expected);

  #if (_XVEC_HAS_CONSTEXPR)
    constexpr auto ceCmplx = TypeParam([](auto i) { return typename TypeParam::value_type(-i, i); });
    constexpr auto ceReal1 = ceCmplx.real();
    constexpr auto ceReal2 = xvec::real(ceCmplx);
    BOOST_SIMD_EQUAL(ceReal1, ceCmplx.real());
    BOOST_SIMD_EQUAL(ceReal2, ceCmplx.real());
  #endif
}

// Set the real values only, leaving the imaginary values unchanged.
BOOST_AUTO_TEST_CASE_TEMPLATE(SetReal, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  // Generate a set of values to write into the reals.
  using ComponentType = decltype(f.v0.real());
  const auto realValues = GetRandomVector<ComponentType>(f.random_limit);

  const auto expected = applyBinary(f.v0, realValues, [](auto c, auto r) { c.real(r); return c; });

  auto computed = f.v0;
  computed.real(realValues);
  BOOST_SIMD_EQUAL(computed, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ExtractImag, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  const auto expected = applyUnary(f.v0, [](auto x) { return x.imag(); });
  BOOST_SIMD_EQUAL(f.v0.imag(), expected);
  BOOST_SIMD_EQUAL(xvec::imag(f.v0), expected);

  #if (_XVEC_HAS_CONSTEXPR)
    constexpr auto ceCmplx = TypeParam([](auto i) { return typename TypeParam::value_type(-i, i); });
    constexpr auto ceReal1 = ceCmplx.imag();
    constexpr auto ceReal2 = xvec::imag(ceCmplx);
    BOOST_SIMD_EQUAL(ceReal1, ceCmplx.imag());
    BOOST_SIMD_EQUAL(ceReal2, ceCmplx.imag());
  #endif
}

// Set the imaginary values only, leaving the real values unchanged.
BOOST_AUTO_TEST_CASE_TEMPLATE(SetImag, TypeParam, ComplexSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // Generate a set of values to write into the imaginaries.
  using ComponentType = decltype(f.v0.real());
  const auto realValues = GetRandomVector<ComponentType>(f.random_limit);

  const auto expected = applyBinary(f.v0, realValues, [](auto c, auto r) { c.imag(r); return c; });

  auto computed = f.v0;
  computed.imag(realValues); // real as in non-complex, not going into `real'.
  BOOST_SIMD_EQUAL(computed, expected);
}
