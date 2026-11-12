#include <xvec/simd>

extern "C" auto xvec_countl_zero_u32(const xvec::simd::vec<std::uint32_t> v)
{
  return countl_zero(v);
}
