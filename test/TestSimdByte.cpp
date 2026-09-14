//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>
#include <boost/test/unit_test.hpp>
#include <cstddef>

using namespace xvec::simd;

const std::byte values[] = {std::byte{0x12}, std::byte{0x34}, std::byte{0x56}, std::byte{0x78}};

BOOST_AUTO_TEST_SUITE(TestSimdByte)

BOOST_AUTO_TEST_CASE(test_to_integer_uint8)
{
  vec<std::byte, 4> b([](auto i) { return values[i]; });

  auto result = to_integer<uint8_t>(b);

  BOOST_CHECK_EQUAL(result[0], std::to_integer<uint8_t>(std::byte{0x12}));
  BOOST_CHECK_EQUAL(result[1], std::to_integer<uint8_t>(std::byte{0x34}));
  BOOST_CHECK_EQUAL(result[2], std::to_integer<uint8_t>(std::byte{0x56}));
  BOOST_CHECK_EQUAL(result[3], std::to_integer<uint8_t>(std::byte{0x78}));
}

BOOST_AUTO_TEST_CASE(test_to_integer_int)
{
  vec<std::byte, 4> b([](auto i) { return values[i]; });
  auto result = to_integer<int>(b);

  BOOST_CHECK_EQUAL(result[0], std::to_integer<int>(std::byte{0x12}));
  BOOST_CHECK_EQUAL(result[1], std::to_integer<int>(std::byte{0x34}));
  BOOST_CHECK_EQUAL(result[2], std::to_integer<int>(std::byte{0x56}));
  BOOST_CHECK_EQUAL(result[3], std::to_integer<int>(std::byte{0x78}));
}

BOOST_AUTO_TEST_SUITE_END()
