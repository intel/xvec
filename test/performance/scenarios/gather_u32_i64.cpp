//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>

#include <xvec/simd>

using Index = xvec::simd::vec<std::int64_t>;

extern "C" auto xvec_gather_u32_i64(const std::array<std::uint32_t, 256>& table, const Index indexes)
{
  return unchecked_gather_from(table, indexes);
}
