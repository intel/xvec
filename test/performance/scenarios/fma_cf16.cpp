//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<_Float16>>;

extern "C" auto xvec_fma_cf16(const Vec x, const Vec y, const Vec acc)
{
  return fma(x, y, acc);
}
