#include <xvec/simd>

using Vec = xvec::simd::vec<std::int32_t>;

extern "C" auto xvec_saturating_add_i32(const Vec a, const Vec b)
{
  return saturating_add(a, b);
}
