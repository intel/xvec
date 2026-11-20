//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<float>>;

extern "C" auto xvec_conj_cf32(const Vec x)
{
  return conj(x);
}
