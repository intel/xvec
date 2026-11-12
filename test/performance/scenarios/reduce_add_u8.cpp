#include <xvec/simd>

extern "C" auto xvec_reduce_add_u8(const xvec::simd::vec<std::uint8_t> v)
{
  return reduce(v);
}
