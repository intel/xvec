//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <xvec/simd>

extern "C" float xvec_inner_product_f16(const xvec::simd::vec<_Float16, 128> x,
                                        const xvec::simd::vec<_Float16, 128> y,
                                        const float init)
{
  return inner_product(x, y, init);
}
