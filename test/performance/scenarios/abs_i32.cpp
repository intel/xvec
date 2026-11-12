#include <xvec/simd>

extern "C" auto xvec_abs_i32(const xvec::simd::vec<std::int32_t> v)
{
  return abs(v);
}
