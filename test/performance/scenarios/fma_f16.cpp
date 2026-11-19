#include <xvec/simd>

extern "C" auto xvec_fma_f16(const xvec::simd::vec<_Float16> x,
                             const xvec::simd::vec<_Float16> y,
                             const xvec::simd::vec<_Float16> acc)
{
  return fma(x, y, acc);
}
