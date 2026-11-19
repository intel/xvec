#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<float>>;

extern "C" auto xvec_eq_cf32(const Vec x, const Vec y)
{
  return x == y;
}
