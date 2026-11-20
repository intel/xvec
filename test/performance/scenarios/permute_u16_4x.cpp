//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint16_t, 4 * xvec::simd::vec<std::uint16_t>::size()>;

extern "C" auto xvec_permute_u16_4x(const Vec v, const Vec indexes)
{
  return permute(v, indexes);
}
