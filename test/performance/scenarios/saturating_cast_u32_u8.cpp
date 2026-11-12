#include <xvec/simd>

extern "C" auto xvec_saturating_cast_u32_u8(const xvec::simd::vec<std::uint32_t> v)
{
  return saturating_cast<std::uint8_t>(v);
}
