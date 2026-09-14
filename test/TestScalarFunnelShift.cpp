//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

/// Reference design for funnel shifts. These aren't part of the standard yet so
/// we need a special test for them until they get added.
namespace ref {

template <typename T>
auto make_combined(T high, T low) {
    constexpr unsigned N = std::numeric_limits<T>::digits;
    std::bitset<2 * N> h(high), l(low);
    return (h << N) | l;
}

template <typename T>
T funnel_shift_right_wide(T high, T low, unsigned s) {
    auto combined = make_combined(high, low);
    combined >>= s;
    const auto allBits = std::bitset<2 * std::numeric_limits<T>::digits>(~T());
    return static_cast<T>((allBits & combined).to_ullong());
}

template <typename T>
T funnel_shift_left_wide(T high, T low, unsigned s) {
    constexpr unsigned N = std::numeric_limits<T>::digits;
    auto combined = make_combined(high, low);
    combined <<= s;
    combined >>= N;
    const auto allBits = std::bitset<2 * std::numeric_limits<T>::digits>(~T());
    return static_cast<T>((allBits & combined).to_ullong());
}

}

using xvec::simd::detail::funnel_shift_left;
using xvec::simd::detail::funnel_shift_right;

// ===================================================================
// 1. IDENTITY: shift == 0
// ===================================================================

BOOST_AUTO_TEST_SUITE(identity_shift_zero)

BOOST_AUTO_TEST_CASE(right_shift_zero_returns_low_u8)  { BOOST_CHECK_EQUAL(funnel_shift_right(uint8_t(0xAB), uint8_t(0xCD), 0), uint8_t(0xCD)); }
BOOST_AUTO_TEST_CASE(right_shift_zero_returns_low_u16) { BOOST_CHECK_EQUAL(funnel_shift_right(uint16_t(0xABCD), uint16_t(0x1234), 0), uint16_t(0x1234)); }
BOOST_AUTO_TEST_CASE(right_shift_zero_returns_low_u32) { BOOST_CHECK_EQUAL(funnel_shift_right(0xDEADBEEFu, 0xCAFEBABEu, 0), 0xCAFEBABEu); }
BOOST_AUTO_TEST_CASE(right_shift_zero_returns_low_u64) { BOOST_CHECK_EQUAL(funnel_shift_right(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL, 0), 0x0123456789ABCDEFULL); }
BOOST_AUTO_TEST_CASE(left_shift_zero_returns_high_u8)  { BOOST_CHECK_EQUAL(funnel_shift_left(uint8_t(0xAB), uint8_t(0xCD), 0), uint8_t(0xAB)); }
BOOST_AUTO_TEST_CASE(left_shift_zero_returns_high_u16) { BOOST_CHECK_EQUAL(funnel_shift_left(uint16_t(0xABCD), uint16_t(0x1234), 0), uint16_t(0xABCD)); }
BOOST_AUTO_TEST_CASE(left_shift_zero_returns_high_u32) { BOOST_CHECK_EQUAL(funnel_shift_left(0xDEADBEEFu, 0xCAFEBABEu, 0), 0xDEADBEEFu); }
BOOST_AUTO_TEST_CASE(left_shift_zero_returns_high_u64) { BOOST_CHECK_EQUAL(funnel_shift_left(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL, 0), 0xDEADBEEFCAFEBABEULL); }

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 2. BOUNDARY: shift == 1
// ===================================================================

BOOST_AUTO_TEST_SUITE(shift_by_one)

BOOST_AUTO_TEST_CASE(right_shift_one_u8)  { BOOST_CHECK_EQUAL(funnel_shift_right(uint8_t(0xFF), uint8_t(0x00), 1), uint8_t(0x80)); }
BOOST_AUTO_TEST_CASE(left_shift_one_u8)   { BOOST_CHECK_EQUAL(funnel_shift_left(uint8_t(0x00), uint8_t(0x80), 1), uint8_t(0x01)); }
BOOST_AUTO_TEST_CASE(right_shift_one_u32) { BOOST_CHECK_EQUAL(funnel_shift_right(1u, 0u, 1), 0x80000000u); }
BOOST_AUTO_TEST_CASE(left_shift_one_u32)  { BOOST_CHECK_EQUAL(funnel_shift_left(0u, 0x80000000u, 1), 1u); }

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 3. BOUNDARY: shift == N-1
// ===================================================================

BOOST_AUTO_TEST_SUITE(shift_by_N_minus_1)

BOOST_AUTO_TEST_CASE(right_u8)  { BOOST_CHECK_EQUAL(funnel_shift_right(uint8_t(0xAB), uint8_t(0xCD), 7),  ref::funnel_shift_right_wide(uint8_t(0xAB), uint8_t(0xCD), 7u)); }
BOOST_AUTO_TEST_CASE(right_u16) { BOOST_CHECK_EQUAL(funnel_shift_right(uint16_t(0xABCD), uint16_t(0x1234), 15), ref::funnel_shift_right_wide(uint16_t(0xABCD), uint16_t(0x1234), 15u)); }
BOOST_AUTO_TEST_CASE(right_u32) { BOOST_CHECK_EQUAL(funnel_shift_right(0xDEADBEEFu, 0x12345678u, 31), ref::funnel_shift_right_wide(0xDEADBEEFu, 0x12345678u, 31u)); }
BOOST_AUTO_TEST_CASE(right_u64) { BOOST_CHECK_EQUAL(funnel_shift_right(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL, 63), ref::funnel_shift_right_wide(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL, 63u)); }
BOOST_AUTO_TEST_CASE(left_u8)   { BOOST_CHECK_EQUAL(funnel_shift_left(uint8_t(0xAB), uint8_t(0xCD), 7),  ref::funnel_shift_left_wide(uint8_t(0xAB), uint8_t(0xCD), 7u)); }
BOOST_AUTO_TEST_CASE(left_u16)  { BOOST_CHECK_EQUAL(funnel_shift_left(uint16_t(0xABCD), uint16_t(0x1234), 15), ref::funnel_shift_left_wide(uint16_t(0xABCD), uint16_t(0x1234), 15u)); }
BOOST_AUTO_TEST_CASE(left_u32)  { BOOST_CHECK_EQUAL(funnel_shift_left(0xDEADBEEFu, 0x12345678u, 31), ref::funnel_shift_left_wide(0xDEADBEEFu, 0x12345678u, 31u)); }
BOOST_AUTO_TEST_CASE(left_u64)  { BOOST_CHECK_EQUAL(funnel_shift_left(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL, 63), ref::funnel_shift_left_wide(0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL, 63u)); }

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 4. PROPOSAL DIAGRAM EXAMPLE
// ===================================================================

BOOST_AUTO_TEST_SUITE(proposal_example)

BOOST_AUTO_TEST_CASE(right_shift_diagram) {
    BOOST_CHECK_EQUAL(funnel_shift_right(uint16_t(0xABCD), uint16_t(0x1234), 8), uint16_t(0xCD12));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 5. ROTATION EQUIVALENCE (boundary shifts only: 0, 1, N/2, N-1)
// ===================================================================

BOOST_AUTO_TEST_SUITE(rotation_equivalence)

template <typename T>
void check_rotation_at(T x, unsigned s) {
    BOOST_CHECK_EQUAL(funnel_shift_right(x, x, s), std::rotr(x, static_cast<int>(s)));
    BOOST_CHECK_EQUAL(funnel_shift_left(x, x, s),  std::rotl(x, static_cast<int>(s)));
}

BOOST_AUTO_TEST_CASE(u8) {
    constexpr unsigned N = 8;
    for (uint8_t x : {uint8_t(0), uint8_t(1), uint8_t(0x55), uint8_t(0xAA), uint8_t(0xFF), uint8_t(0x80)})
        for (unsigned s : {0u, 1u, N / 2, N - 1})
            check_rotation_at(x, s);
}
BOOST_AUTO_TEST_CASE(u32) {
    constexpr unsigned N = 32;
    for (uint32_t x : {0u, 1u, 0x55555555u, 0xAAAAAAAAu, 0xFFFFFFFFu, 0x80000000u, 0xDEADBEEFu})
        for (unsigned s : {0u, 1u, N / 2, N - 1})
            check_rotation_at(x, s);
}
BOOST_AUTO_TEST_CASE(u64) {
    constexpr unsigned N = 64;
    for (uint64_t x : {0ULL, 1ULL, 0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL, 0xFFFFFFFFFFFFFFFFULL, 0x8000000000000000ULL})
        for (unsigned s : {0u, 1u, N / 2, N - 1})
            check_rotation_at(x, s);
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 6. ALL-ZERO AND ALL-ONE INPUTS (boundary shifts only)
// ===================================================================

BOOST_AUTO_TEST_SUITE(special_values)

BOOST_AUTO_TEST_CASE(both_zero_always_zero) {
    for (unsigned s : {0u, 1u, 15u, 31u}) {
        BOOST_CHECK_EQUAL(funnel_shift_right(0u, 0u, s), 0u);
        BOOST_CHECK_EQUAL(funnel_shift_left(0u, 0u, s), 0u);
    }
}
BOOST_AUTO_TEST_CASE(both_ones_always_ones) {
    constexpr uint32_t ones = 0xFFFFFFFF;
    for (unsigned s : {0u, 1u, 15u, 31u}) {
        BOOST_CHECK_EQUAL(funnel_shift_right(ones, ones, s), ones);
        BOOST_CHECK_EQUAL(funnel_shift_left(ones, ones, s), ones);
    }
}
BOOST_AUTO_TEST_CASE(high_zero_low_ones) {
    for (unsigned s : {0u, 1u, 8u, 15u}) {
        BOOST_CHECK_EQUAL(funnel_shift_right(uint16_t(0), uint16_t(0xFFFF), s), ref::funnel_shift_right_wide(uint16_t(0), uint16_t(0xFFFF), s));
        BOOST_CHECK_EQUAL(funnel_shift_left(uint16_t(0), uint16_t(0xFFFF), s),  ref::funnel_shift_left_wide(uint16_t(0), uint16_t(0xFFFF), s));
    }
}
BOOST_AUTO_TEST_CASE(high_ones_low_zero) {
    for (unsigned s : {0u, 1u, 8u, 15u}) {
        BOOST_CHECK_EQUAL(funnel_shift_right(uint16_t(0xFFFF), uint16_t(0), s), ref::funnel_shift_right_wide(uint16_t(0xFFFF), uint16_t(0), s));
        BOOST_CHECK_EQUAL(funnel_shift_left(uint16_t(0xFFFF), uint16_t(0), s),  ref::funnel_shift_left_wide(uint16_t(0xFFFF), uint16_t(0), s));
    }
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 7. EXHAUSTIVE u8 (single pass, plain comparisons in hot loop)
//    + SAMPLED u16/u32/u64
//
// The u8 loop does all comparisons with plain ==, accumulating a
// failure count. Only one BOOST_CHECK_EQUAL fires per test case,
// comparing failure count to 0. This avoids millions of Boost
// assertion calls that dominate runtime.
// ===================================================================

BOOST_AUTO_TEST_SUITE(cross_check_wide)

BOOST_AUTO_TEST_CASE(exhaustive_u8) {
    unsigned failures = 0;
    for (unsigned h = 0; h < 256; ++h) {
        for (unsigned l = 0; l < 256; ++l) {
            auto hi = uint8_t(h), lo = uint8_t(l);
            for (unsigned s = 0; s < 8; ++s) {
                if (funnel_shift_right(hi, lo, s) !=
                    ref::funnel_shift_right_wide(hi, lo, s))
                    ++failures;
                if (funnel_shift_left(hi, lo, s) !=
                    ref::funnel_shift_left_wide(hi, lo, s))
                    ++failures;
                if (s > 0 && funnel_shift_right(hi, lo, s) !=
                             funnel_shift_left(hi, lo, 8u - s))
                    ++failures;
                if (h == l) {
                    if (funnel_shift_right(hi, lo, s) !=
                        std::rotr(hi, static_cast<int>(s)))
                        ++failures;
                    if (funnel_shift_left(hi, lo, s) !=
                        std::rotl(hi, static_cast<int>(s)))
                        ++failures;
                }
            }
        }
    }
    BOOST_CHECK_EQUAL(failures, 0u);
}

BOOST_AUTO_TEST_CASE(sampled_u16) {
    const uint16_t vals[] = {0x0000, 0x0001, 0x5555, 0x8000, 0xABCD, 0xFFFF};
    const unsigned shifts[] = {0, 1, 7, 8, 15};
    for (auto h : vals)
        for (auto l : vals)
            for (auto s : shifts) {
                BOOST_CHECK_EQUAL(funnel_shift_right(h, l, s), ref::funnel_shift_right_wide(h, l, s));
                BOOST_CHECK_EQUAL(funnel_shift_left(h, l, s),  ref::funnel_shift_left_wide(h, l, s));
            }
}

BOOST_AUTO_TEST_CASE(sampled_u32) {
    const uint32_t vals[] = {0u, 1u, 0x55555555u, 0x80000000u, 0xDEADBEEFu, 0xFFFFFFFFu};
    const unsigned shifts[] = {0, 1, 8, 16, 24, 31};
    for (auto h : vals)
        for (auto l : vals)
            for (auto s : shifts) {
                BOOST_CHECK_EQUAL(funnel_shift_right(h, l, s), ref::funnel_shift_right_wide(h, l, s));
                BOOST_CHECK_EQUAL(funnel_shift_left(h, l, s),  ref::funnel_shift_left_wide(h, l, s));
            }
}

BOOST_AUTO_TEST_CASE(sampled_u64) {
    const uint64_t vals[] = {0ULL, 1ULL, 0x5555555555555555ULL, 0x8000000000000000ULL, 0xDEADBEEFCAFEBABEULL, 0xFFFFFFFFFFFFFFFFULL};
    const unsigned shifts[] = {0, 1, 16, 32, 48, 63};
    for (auto h : vals)
        for (auto l : vals)
            for (auto s : shifts) {
                BOOST_CHECK_EQUAL(funnel_shift_right(h, l, s), ref::funnel_shift_right_wide(h, l, s));
                BOOST_CHECK_EQUAL(funnel_shift_left(h, l, s),  ref::funnel_shift_left_wide(h, l, s));
            }
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 8. SINGLE BIT PROPAGATION
// ===================================================================

BOOST_AUTO_TEST_SUITE(single_bit)

BOOST_AUTO_TEST_CASE(single_bit_in_high_right_u32) {
    for (unsigned s = 1; s < 32; ++s)
        BOOST_CHECK_EQUAL(funnel_shift_right(1u, 0u, s), 1u << (32u - s));
}
BOOST_AUTO_TEST_CASE(single_bit_in_low_left_u32) {
    for (unsigned s = 1; s < 32; ++s)
        BOOST_CHECK_EQUAL(funnel_shift_left(0u, 0x80000000u, s), 1u << (s - 1u));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 9. constexpr VERIFICATION
// ===================================================================

BOOST_AUTO_TEST_SUITE(constexpr_checks)

BOOST_AUTO_TEST_CASE(constexpr_right) {
    constexpr uint32_t r = funnel_shift_right(0xDEADBEEFu, 0x12345678u, 16);
    static_assert(r == 0xBEEF1234u);
    BOOST_CHECK_EQUAL(r, 0xBEEF1234u);
}
BOOST_AUTO_TEST_CASE(constexpr_left) {
    constexpr uint32_t r = funnel_shift_left(0xDEADBEEFu, 0x12345678u, 16);
    BOOST_CHECK_EQUAL(r, ref::funnel_shift_left_wide(0xDEADBEEFu, 0x12345678u, 16u));
}
BOOST_AUTO_TEST_CASE(constexpr_zero_shift) {
    static_assert(funnel_shift_right(uint16_t(0xABCD), uint16_t(0x1234), 0) == 0x1234);
    static_assert(funnel_shift_left(uint16_t(0xABCD), uint16_t(0x1234), 0) == 0xABCD);
    BOOST_CHECK(true);
}
BOOST_AUTO_TEST_CASE(constexpr_proposal_example) {
    static_assert(funnel_shift_right(uint16_t(0xABCD), uint16_t(0x1234), 8) == 0xCD12);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 10. SHIFT-COUNT TYPE VARIANTS
// ===================================================================

BOOST_AUTO_TEST_SUITE(shift_count_types)

BOOST_AUTO_TEST_CASE(shift_int)      { BOOST_CHECK_EQUAL(funnel_shift_right(0xFFu, 0u, int(4)),      ref::funnel_shift_right_wide(0xFFu, 0u, 4u)); }
BOOST_AUTO_TEST_CASE(shift_unsigned) { BOOST_CHECK_EQUAL(funnel_shift_right(0xFFu, 0u, unsigned(4)), ref::funnel_shift_right_wide(0xFFu, 0u, 4u)); }
BOOST_AUTO_TEST_CASE(shift_short)    { BOOST_CHECK_EQUAL(funnel_shift_right(0xFFu, 0u, short(4)),    ref::funnel_shift_right_wide(0xFFu, 0u, 4u)); }
BOOST_AUTO_TEST_CASE(shift_int64)    { BOOST_CHECK_EQUAL(funnel_shift_right(0xFFu, 0u, int64_t(4)),  ref::funnel_shift_right_wide(0xFFu, 0u, 4u)); }
BOOST_AUTO_TEST_CASE(shift_uint8)    { BOOST_CHECK_EQUAL(funnel_shift_right(0xFFu, 0u, uint8_t(4)),  ref::funnel_shift_right_wide(0xFFu, 0u, 4u)); }

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 11. HASH MIXING EXAMPLE
// ===================================================================

BOOST_AUTO_TEST_SUITE(hash_mixing_example)

BOOST_AUTO_TEST_CASE(hash_mix) {
    auto mix = [](uint32_t x, uint32_t y) { return funnel_shift_right(x, y, 17); };
    BOOST_CHECK_EQUAL(mix(0xDEADBEEFu, 0xCAFEBABEu),
                      ref::funnel_shift_right_wide(0xDEADBEEFu, 0xCAFEBABEu, 17u));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 12. TYPE CONSTRAINTS (SFINAE)
// ===================================================================

BOOST_AUTO_TEST_SUITE(type_constraints)

namespace detail {
template <typename T, typename S, typename = void>
struct has_fsr : std::false_type {};
template <typename T, typename S>
struct has_fsr<T, S, std::void_t<decltype(funnel_shift_right(std::declval<T>(), std::declval<T>(), std::declval<S>()))>> : std::true_type {};
template <typename T, typename S, typename = void>
struct has_fsl : std::false_type {};
template <typename T, typename S>
struct has_fsl<T, S, std::void_t<decltype(funnel_shift_left(std::declval<T>(), std::declval<T>(), std::declval<S>()))>> : std::true_type {};
}

BOOST_AUTO_TEST_CASE(unsigned_accepted) {
    BOOST_CHECK((detail::has_fsr<uint8_t,  int>::value));
    BOOST_CHECK((detail::has_fsr<uint16_t, int>::value));
    BOOST_CHECK((detail::has_fsr<uint32_t, int>::value));
    BOOST_CHECK((detail::has_fsr<uint64_t, int>::value));
}
BOOST_AUTO_TEST_CASE(signed_rejected) {
    BOOST_CHECK((!detail::has_fsr<int8_t,  int>::value));
    BOOST_CHECK((!detail::has_fsr<int32_t, int>::value));
    BOOST_CHECK((!detail::has_fsl<int64_t, int>::value));
}
BOOST_AUTO_TEST_CASE(bool_rejected) {
    BOOST_CHECK((!detail::has_fsr<bool, int>::value));
    BOOST_CHECK((!detail::has_fsl<bool, int>::value));
}
BOOST_AUTO_TEST_CASE(float_shift_rejected) {
    BOOST_CHECK((!detail::has_fsr<uint32_t, float>::value));
    BOOST_CHECK((!detail::has_fsl<uint32_t, double>::value));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 13. NOEXCEPT
// ===================================================================

BOOST_AUTO_TEST_SUITE(noexcept_check)

BOOST_AUTO_TEST_CASE(all_noexcept) {
    static_assert(noexcept(funnel_shift_right(0u, 0u, 0)));
    static_assert(noexcept(funnel_shift_left(0u, 0u, 0)));
    static_assert(noexcept(funnel_shift_right(uint64_t(0), uint64_t(0), 0)));
    static_assert(noexcept(funnel_shift_left(uint64_t(0), uint64_t(0), 0)));
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 14. BIT-FIELD EXTRACTION EXAMPLE
// ===================================================================

BOOST_AUTO_TEST_SUITE(bit_extraction_example)

BOOST_AUTO_TEST_CASE(extract_bits_across_boundary) {
    uint64_t data[] = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
    auto extract = [&](size_t bit_offset) -> uint64_t {
        size_t wi = bit_offset / 64, bi = bit_offset % 64;
        if (bi == 0) return data[wi];
        return funnel_shift_right(data[wi + 1], data[wi], static_cast<int>(bi));
    };

    BOOST_CHECK_EQUAL(extract(0), data[0]);
    for (unsigned bi : {1u, 16u, 32u, 48u, 63u})
        BOOST_CHECK_EQUAL(extract(bi), ref::funnel_shift_right_wide(data[1], data[0], bi));
}

BOOST_AUTO_TEST_SUITE_END()
