#include <xvec/simd>

extern "C" auto xvec_abs_i8(const xvec::simd::vec<std::int8_t> v)
{
  return abs(v);
}
