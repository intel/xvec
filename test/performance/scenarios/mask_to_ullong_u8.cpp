//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

extern "C" auto xvec_mask_to_ullong_u8(const xvec::simd::vec<std::uint8_t>::mask_type m)
{
  return m.to_ullong();
}
