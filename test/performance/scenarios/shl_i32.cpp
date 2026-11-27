//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

using Vec = xvec::simd::vec<std::int32_t>;

extern "C" auto xvec_shl_i32(Vec v, Vec counts)
{
  return shl(v, counts);
}
