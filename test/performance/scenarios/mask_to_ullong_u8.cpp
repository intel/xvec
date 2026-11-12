#include <xvec/simd>

extern "C" auto xvec_mask_to_ullong_u8(const xvec::simd::vec<std::uint8_t>::mask_type m)
{
  return m.to_ullong();
}
