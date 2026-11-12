#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint32_t>;

extern "C" auto xvec_compress_u32(const Vec v, const Vec::mask_type m)
{
  return compress(v, m);
}
