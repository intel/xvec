#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<double>, 2 * xvec::simd::vec<std::complex<double>>::size()>;

extern "C" auto xvec_compress_complex_f64_2x(const Vec v, const Vec::mask_type m)
{
  return compress(v, m);
}
