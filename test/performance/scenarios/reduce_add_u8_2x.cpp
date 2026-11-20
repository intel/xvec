//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint8_t, 2 * xvec::simd::vec<std::uint8_t>::size()>;

extern "C" auto xvec_reduce_add_u8_2x(const Vec v)
{
  return reduce(v);
}
