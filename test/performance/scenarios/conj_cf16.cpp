#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<_Float16>>;

extern "C" auto xvec_conj_cf16(const Vec x)
{
  return conj(x);
}
