//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<double>>;
using Index = xvec::simd::vec<std::int64_t, Vec::size()>;

extern "C" auto xvec_gather_complex_f64(const std::array<std::complex<double>, 256>& table,
                                        const Index indexes)
{
  return unchecked_gather_from(table, indexes);
}
