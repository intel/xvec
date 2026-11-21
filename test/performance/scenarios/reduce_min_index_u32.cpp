//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

extern "C" auto xvec_reduce_min_index_u32(const xvec::simd::vec<std::uint32_t>::mask_type m)
{
  return reduce_min_index(m);
}
