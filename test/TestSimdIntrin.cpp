//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

#if defined(__GFNI__) && !defined(_XVEC_FORCE_SCALAR)

using namespace xvec::simd::x86;

using Uint8SimdTypes = boost::mp11::mp_product<to_vec, boost::mp11::mp_list<uint8_t>, AllSizes>;
BOOST_AUTO_TEST_CASE_TEMPLATE(Gfni, TypeParam, Uint8SimdTypes)
{
  SimdTestFixture<TypeParam> f;

  // This GFNI control word reverse the bits in a byte.
  uint64_t revBits = 0x8040201008040201;

  // Scalar call reference implementation - reverse and invert the bits.
  auto ref = [](uint8_t b) -> uint8_t 
  { uint8_t r = 0;
    for (int i=0; i<8; ++i)
      r = (r << 1) | ((b >> i) & 1);
    return r ^ 0xff;
  };

  // The raw form requires exact multiples of 8 bytes to work properly, and is
  // then called with one matrix element for each block of 8.
  constexpr int roundedSize = (TypeParam::size() + 7) & ~7;
  auto as64 = grow<roundedSize>(f.v0);
  auto m64 = xvec::simd::vec<uint64_t, roundedSize / 8>(revBits);
  auto computedRaw = xvec::simd::x86::gf2p8affine<0xff>(as64, m64);
  auto expectedRaw = applyUnary(as64, [&](auto b) -> uint8_t { return ref(b); });
  BOOST_SIMD_EQUAL(computedRaw, expectedRaw);

  // The refined form takes a single 64-bit matrix and applies it to however
  // many bytes are provided.
  auto expectedRefined = applyUnary(f.v0, [&](auto b) -> uint8_t { return ref(b); });
  auto computedRefined = xvec::simd::x86::gf2p8affine<0xff>(f.v0, revBits);
  BOOST_SIMD_EQUAL(computedRefined, expectedRefined);
}
#else
// On machines without GFNI then this test file will be empty. Provide at least
// something, so that boost doesn't complain of an empty test.
BOOST_AUTO_TEST_CASE(GfniUnavailable)
{
  BOOST_TEST(true);
}
#endif


