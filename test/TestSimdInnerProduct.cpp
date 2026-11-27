//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

#if defined(__FLT16_MIN__)
using Fp16SimdTypes = to_simd<Fp16FloatTypes>;

BOOST_AUTO_TEST_CASE_TEMPLATE(Fp16InnerProduct, TypeParam, Fp16SimdTypes)
{
  // Small integer inputs keep every partial sum exact in FP32, in any order.
  const TypeParam lhs = GetRandomVector<TypeParam>(8);
  const TypeParam rhs = GetRandomVector<TypeParam>(8);

  float expected = 0.0f;
  for (int i = 0; i < TypeParam::size(); ++i)
    expected += float(lhs[i]) * float(rhs[i]);

  BOOST_TEST(xvec::simd::inner_product(lhs, rhs) == expected);
  BOOST_TEST(xvec::simd::inner_product(lhs, rhs, 100.0f) == expected + 100.0f);
}

// An even size that is not a multiple of the register size exercises the partial final chunk.
BOOST_AUTO_TEST_CASE(Fp16InnerProductPartialChunk)
{
  using Vec = xvec::simd::vec<_Float16, 34>;
  const Vec lhs = GetRandomVector<Vec>(8);
  const Vec rhs = GetRandomVector<Vec>(8);

  float expected = 0.0f;
  for (int i = 0; i < Vec::size(); ++i)
    expected += float(lhs[i]) * float(rhs[i]);

  BOOST_TEST(xvec::simd::inner_product(lhs, rhs, 100.0f) == expected + 100.0f);
}

// Products and sums are formed in FP32, so values that would overflow or
// stall in FP16 arithmetic must still give exact results.
BOOST_AUTO_TEST_CASE_TEMPLATE(Fp16InnerProductAccumulatesInFp32, TypeParam, Fp16SimdTypes)
{
  // 300 * 300 exceeds the largest FP16 value (65504) when multiplied.
  const TypeParam big(300.f16);
  BOOST_TEST(xvec::simd::inner_product(big, big) == 90000.0f * TypeParam::size());

  // 2048 + 1 is not representable in FP16, so FP16 accumulation would stall.
  const TypeParam ones(1.f16);
  BOOST_TEST(xvec::simd::inner_product(ones, ones, 2048.0f) == 2048.0f + TypeParam::size());
}

#if defined(_XVEC_HAS_CONSTEXPR)
constexpr float constexprInnerProduct()
{
  using Vec = xvec::simd::vec<_Float16, 5>;
  const Vec lhs = xvec::simd::iota<Vec> + 1.f16;
  const Vec rhs = 2.f16 * xvec::simd::iota<Vec> - 3.f16;
  return xvec::simd::inner_product(lhs, rhs, 10.0f);
}

// 10 + (1*-3) + (2*-1) + (3*1) + (4*3) + (5*5) = 45
static_assert(constexprInnerProduct() == 45.0f);
#endif

#else // __FLT16_MIN__

// Keep the suite non-empty on targets without FP16 support.
BOOST_AUTO_TEST_CASE(Fp16Unavailable)
{
  BOOST_TEST(true);
}
#endif // __FLT16_MIN__
