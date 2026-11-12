#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint32_t, 4 * xvec::simd::vec<std::uint32_t>::size()>;

extern "C" auto xvec_permute_u32_4x(const Vec v, const Vec indexes)
{
  return permute(v, indexes);
}
