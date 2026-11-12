#include <array>
#include <complex>

#include <xvec/simd>

using Vec = xvec::simd::vec<std::complex<double>>;
using Index = xvec::simd::vec<std::int64_t, Vec::size()>;

extern "C" void xvec_scatter_complex_f64(const Vec values,
                                         std::array<std::complex<double>, 256>& table,
                                         const Index indexes)
{
  unchecked_scatter_to(values, table, indexes);
}
