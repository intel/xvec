#include <array>

#include <xvec/simd>

extern "C" void xvec_scatter_u32(const xvec::simd::vec<std::uint32_t> values,
                                 std::array<std::uint32_t, 256>& table,
                                 const xvec::simd::vec<std::int32_t> indexes)
{
  unchecked_scatter_to(values, table, indexes);
}
