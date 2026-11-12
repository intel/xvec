#include <xvec/simd>

extern "C" auto xvec_bit_reverse_u8(const xvec::simd::vec<std::uint8_t> v)
{
  return bit_reverse(v);
}
