#include <xvec/simd>

using Mask = xvec::simd::vec<std::uint8_t>::mask_type;

extern "C" auto xvec_mask_from_integer_u8(const std::uint64_t bits)
{
  return Mask(bits);
}
