#include <xvec/simd>

extern "C" auto xvec_abs_i64(const xvec::simd::vec<std::int64_t> v)
{
  return abs(v);
}
