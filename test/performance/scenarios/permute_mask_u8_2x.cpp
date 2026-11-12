#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint8_t, 2 * xvec::simd::vec<std::uint8_t>::size()>;

extern "C" auto xvec_permute_mask_u8_2x(const Vec::mask_type m, const Vec indexes)
{
  return permute(m, indexes);
}
