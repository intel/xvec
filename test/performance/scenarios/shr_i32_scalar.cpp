//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

using Vec = xvec::simd::vec<std::int32_t>;

extern "C" auto xvec_shr_i32_scalar(Vec v, std::int32_t count)
{
  return shr(v, count);
}
