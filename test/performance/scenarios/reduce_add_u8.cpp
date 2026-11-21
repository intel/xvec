//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

extern "C" auto xvec_reduce_add_u8(const xvec::simd::vec<std::uint8_t> v)
{
  return reduce(v);
}
