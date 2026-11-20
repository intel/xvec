#include <array>

#include <xvec/simd>

using Index = xvec::simd::vec<std::int32_t, xvec::simd::vec<std::uint64_t>::size()>;

extern "C" auto xvec_gather_u64_i32(const std::array<std::uint64_t, 256>& table, const Index indexes)
{
  return unchecked_gather_from(table, indexes);
}
