//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<double>>;
using Index = xvec::simd::vec<std::uint64_t, Vec::size()>;

extern "C" auto xvec_permute_complex_f64(const Vec v, const Index indexes)
{
  return permute(v, indexes);
}
