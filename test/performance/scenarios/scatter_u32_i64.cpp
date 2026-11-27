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
using Values = xvec::simd::vec<std::uint32_t, Index::size()>;

extern "C" void xvec_scatter_u32_i64(const Values values, std::array<std::uint32_t, 256>& table, const Index indexes)
{
  unchecked_scatter_to(values, table, indexes);
}
