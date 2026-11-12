#include <array>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint32_t>;

extern "C" auto xvec_load_masked_u32(const std::array<std::uint32_t, 64>& data, const Vec::mask_type m)
{
  return xvec::simd::partial_load<Vec>(data, m);
}
