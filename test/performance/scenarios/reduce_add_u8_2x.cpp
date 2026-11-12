#include <xvec/simd>

using Vec = xvec::simd::vec<std::uint8_t, 2 * xvec::simd::vec<std::uint8_t>::size()>;

extern "C" auto xvec_reduce_add_u8_2x(const Vec v)
{
  return reduce(v);
}
