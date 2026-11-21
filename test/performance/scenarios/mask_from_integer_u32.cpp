//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

using Mask = xvec::simd::vec<std::uint32_t>::mask_type;

extern "C" auto xvec_mask_from_integer_u32(const std::uint64_t bits)
{
  return Mask(bits);
}
