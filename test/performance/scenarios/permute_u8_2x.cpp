#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint8_t, 2 * xvec::simd::vec<std::uint8_t>::size()>;

extern "C" auto xvec_permute_u8_2x(const Vec v, const Vec indexes)
{
  return permute(v, indexes);
}
