#include <array>

#include <xvec/simd>

extern "C" auto xvec_gather_u32(const std::array<std::uint32_t, 256>& table,
                                const xvec::simd::vec<std::int32_t> indexes)
{
  return unchecked_gather_from(table, indexes);
}
