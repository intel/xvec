//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that these representative operations need no external runtime symbols.
#if defined(__GNUC__) && !defined(__clang__)
#define _MM_MALLOC_H_INCLUDED // GCC's mm_malloc.h needs hosted malloc/free.
#endif

#include <xvec/simd>

using namespace xvec::simd;

auto freestanding_probe(const vec<float, 8>& a, const vec<float, 8>& b, float* p)
{
  auto r = sqrt(a * a + b * b);
  unchecked_store(r, std::span<float, 8>(p, 8));
  return reduce(r);
}

// A user-defined complex type works without <complex>.
struct MyComplex
{
  float re, im;
  MyComplex(float r, float i) : re(r), im(i) {}
  float real() const { return re; }
  float imag() const { return im; }
  friend MyComplex operator+(MyComplex a, MyComplex b)
    { return MyComplex(a.re + b.re, a.im + b.im); }
};
template<> struct xvec::simd::complex_traits<MyComplex> { using value_type = float; };

auto freestanding_complex_probe(const vec<MyComplex, 4>& a, const vec<MyComplex, 4>& b)
{
  return conj(a + b);
}
