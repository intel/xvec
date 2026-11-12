#include <xvec/simd>

extern "C" auto xvec_popcount_u32x16(const xvec::simd::vec<std::uint32_t> v)
{
  return popcount(v);
}
