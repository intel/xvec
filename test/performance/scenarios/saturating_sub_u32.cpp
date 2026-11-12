#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint32_t>;

extern "C" auto xvec_saturating_sub_u32(const Vec a, const Vec b)
{
  return saturating_sub(a, b);
}
