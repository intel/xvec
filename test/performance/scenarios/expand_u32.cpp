//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint32_t>;

extern "C" auto xvec_expand_u32(const Vec v, const Vec::mask_type m)
{
  return expand(v, m);
}
