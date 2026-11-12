#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint32_t, 4 * xvec::simd::vec<std::uint32_t>::size()>;

extern "C" auto xvec_permute_mask_u32_4x(const Vec::mask_type m, const Vec indexes)
{
  return permute(m, indexes);
}
