//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test_funnel_shift_simd.cpp
//
// Boost.Test suite for the SIMD funnel_shift_{left,right} overloads
// as specified in P4010R0.
//
// Uses test suite utilities: applyBinary, applyTernary, iota<VP>,
// BOOST_SIMD_EQUAL.

#include <boost/test/unit_test.hpp>

#include "SimdTestUtilities.hpp"

#include <array>
#include <bit>
#include <bitset>
#include <cstdint>
#include <limits>
#include <tuple>

using xvec::simd::vec;
using xvec::simd::basic_vec;
using xvec::simd::iota;

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
T fsr(T high, T low, unsigned s) {
    auto combined = make_combined(high, low);
    combined >>= s;
    const auto allBits = std::bitset<2 * std::numeric_limits<T>::digits>(~T());
    return static_cast<T>((allBits & combined).to_ullong());
}

template <typename T>
T fsl(T high, T low, unsigned s) {
    constexpr unsigned N = std::numeric_limits<T>::digits;
    auto combined = make_combined(high, low);
    combined <<= s;
    combined >>= N;
    const auto allBits = std::bitset<2 * std::numeric_limits<T>::digits>(~T());
    return static_cast<T>((allBits & combined).to_ullong());
}

}
// ---------------------------------------------------------------------------
// Generate N distinct test values for type T.
// ---------------------------------------------------------------------------
template <typename T, std::size_t N>
constexpr std::array<T, N> generate_values(uint64_t seed) {
    std::array<T, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        uint64_t x = seed ^ (i * 0x9E3779B97F4A7C15ULL);
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        result[i] = static_cast<T>(x);
    }
    constexpr T allOnes = std::numeric_limits<T>::max();
    if constexpr (N >= 8) {
        result[0] = T(0);
        result[1] = allOnes;
        result[2] = static_cast<T>(T(1) << (std::numeric_limits<T>::digits - 1));
        result[3] = T(1);
        result[4] = static_cast<T>(T(0x55) * (allOnes / T(0xFF)));
        result[5] = static_cast<T>(T(0xAA) * (allOnes / T(0xFF)));
        result[6] = static_cast<T>(allOnes >> (std::numeric_limits<T>::digits / 2));
        result[7] = static_cast<T>(allOnes << (std::numeric_limits<T>::digits / 2));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Per-type traits: N lanes = N bits, so iota covers all valid shifts.
// Shift vec uses the same element type T (not unsigned) to satisfy the
// same-size constraint for per-element shifts.
// ---------------------------------------------------------------------------
template <typename T> struct TestTraits {
    static constexpr unsigned N = std::numeric_limits<T>::digits;
    using VT = vec<T, N>;
    using VS = vec<T, N>;  // same element type as value

    static constexpr auto highs() { return generate_values<T, N>(0xDEADBEEFULL); }
    static constexpr auto lows()  { return generate_values<T, N>(0xCAFEBABEULL); }
};

using UnsignedIntegers = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t>;

// ===================================================================
// 1. PER-ELEMENT SHIFT via iota: one call tests all N valid shifts
//    Shift vec is simd<T, N> to satisfy the same-size constraint.
// ===================================================================

BOOST_AUTO_TEST_SUITE(per_element_iota)

BOOST_AUTO_TEST_CASE_TEMPLATE(right, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    auto shifts = iota<typename Tr::VS>;
    BOOST_SIMD_EQUAL(
        funnel_shift_right(high, low, shifts),
        applyTernary(high, low, shifts,
            [](T h, T l, T s) -> T { return ref::fsr(h, l, static_cast<unsigned>(s)); }));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(left, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    auto shifts = iota<typename Tr::VS>;
    BOOST_SIMD_EQUAL(
        funnel_shift_left(high, low, shifts),
        applyTernary(high, low, shifts,
            [](T h, T l, T s) -> T { return ref::fsl(h, l, static_cast<unsigned>(s)); }));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 2. UNIFORM SHIFT: scalar shift (any integral type), all N shifts
// ===================================================================

BOOST_AUTO_TEST_SUITE(uniform_all_shifts)

BOOST_AUTO_TEST_CASE_TEMPLATE(right, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    for (unsigned s = 0; s < Tr::N; ++s)
        BOOST_SIMD_EQUAL(
            funnel_shift_right(high, low, s),
            applyBinary(high, low,
                [s](T h, T l) -> T { return ref::fsr(h, l, s); }));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(left, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    for (unsigned s = 0; s < Tr::N; ++s)
        BOOST_SIMD_EQUAL(
            funnel_shift_left(high, low, s),
            applyBinary(high, low,
                [s](T h, T l) -> T { return ref::fsl(h, l, s); }));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 3. IDENTITY: shift==0
// ===================================================================

BOOST_AUTO_TEST_SUITE(identity_shift_zero)

BOOST_AUTO_TEST_CASE_TEMPLATE(right_returns_low, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    BOOST_SIMD_EQUAL(funnel_shift_right(high, low, 0), low);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(left_returns_high, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    BOOST_SIMD_EQUAL(funnel_shift_left(high, low, 0), high);
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 4. LEFT-RIGHT INVERSE: right(h,l,s) == left(h,l,N-s)
// ===================================================================

BOOST_AUTO_TEST_SUITE(left_right_inverse)

BOOST_AUTO_TEST_CASE_TEMPLATE(inverse, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    for (unsigned s = 1; s < Tr::N; ++s)
        BOOST_SIMD_EQUAL(
            funnel_shift_right(high, low, s),
            funnel_shift_left(high, low, Tr::N - s));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 5. ROTATION: funnel_shift_{right,left}(x,x,s) == rot{r,l}(x,s)
//    Uses per-element iota shift (same type T).
// ===================================================================

BOOST_AUTO_TEST_SUITE(rotation_equivalence)

BOOST_AUTO_TEST_CASE_TEMPLATE(right_equals_rotr, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec v = Tr::highs();
    auto shifts = iota<typename Tr::VS>;
    BOOST_SIMD_EQUAL(
        funnel_shift_right(v, v, shifts),
        applyBinary(v, shifts,
            [](T x, T s) -> T { return std::rotr(x, static_cast<int>(s)); }));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(left_equals_rotl, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    basic_vec v = Tr::highs();
    auto shifts = iota<typename Tr::VS>;
    BOOST_SIMD_EQUAL(
        funnel_shift_left(v, v, shifts),
        applyBinary(v, shifts,
            [](T x, T s) -> T { return std::rotl(x, static_cast<int>(s)); }));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 6. ALL-ZEROS / ALL-ONES invariants
//    Uses per-element iota shift (same type T).
// ===================================================================

BOOST_AUTO_TEST_SUITE(special_values)

BOOST_AUTO_TEST_CASE_TEMPLATE(both_zero_always_zero, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    typename Tr::VT zero(T(0));
    auto shifts = iota<typename Tr::VS>;
    BOOST_SIMD_EQUAL(funnel_shift_right(zero, zero, shifts), zero);
    BOOST_SIMD_EQUAL(funnel_shift_left(zero, zero, shifts), zero);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(both_ones_always_ones, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    typename Tr::VT ones(std::numeric_limits<T>::max());
    auto shifts = iota<typename Tr::VS>;
    BOOST_SIMD_EQUAL(funnel_shift_right(ones, ones, shifts), ones);
    BOOST_SIMD_EQUAL(funnel_shift_left(ones, ones, shifts), ones);
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 7. SIGNED SHIFT TYPE: per-element shift with signed equivalent
//    Verifies that e.g. simd<int16_t> works as the shift type for
//    simd<uint16_t> values (same size, signed).
// ===================================================================

BOOST_AUTO_TEST_SUITE(signed_shift_type)

BOOST_AUTO_TEST_CASE_TEMPLATE(right_signed_shift, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    using ST = std::make_signed_t<T>;
    using VSS = vec<ST, Tr::N>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    // Build signed shift vector with values [0..N-1]
    auto shifts = iota<VSS>;
    BOOST_SIMD_EQUAL(
        funnel_shift_right(high, low, shifts),
        applyTernary(high, low, shifts,
            [](T h, T l, ST s) -> T { return ref::fsr(h, l, static_cast<unsigned>(s)); }));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(left_signed_shift, T, UnsignedIntegers) {
    using Tr = TestTraits<T>;
    using ST = std::make_signed_t<T>;
    using VSS = vec<ST, Tr::N>;
    basic_vec high = Tr::highs();
    basic_vec low  = Tr::lows();
    auto shifts = iota<VSS>;
    BOOST_SIMD_EQUAL(
        funnel_shift_left(high, low, shifts),
        applyTernary(high, low, shifts,
            [](T h, T l, ST s) -> T { return ref::fsl(h, l, static_cast<unsigned>(s)); }));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 8. PROPOSAL DIAGRAM EXAMPLE (uint16_t-specific)
// ===================================================================

BOOST_AUTO_TEST_SUITE(proposal_example)

BOOST_AUTO_TEST_CASE(diagram_u16) {
    vec<uint16_t, 1> high(uint16_t(0xABCD)), low(uint16_t(0x1234));
    auto got = funnel_shift_right(high, low, 8);
    BOOST_CHECK_EQUAL(got[0], uint16_t(0xCD12));
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 9. MIXED-CASES-IN-ONE-VEC (uint16_t-specific)
//    16 hand-picked triples covering identity, boundaries, diagram
//    example, single-bit, alternating patterns, rotation — one call.
//    Shift vec is uint16_t to match value type.
// ===================================================================

BOOST_AUTO_TEST_SUITE(u16_mixed_cases)

BOOST_AUTO_TEST_CASE(right_mixed) {
    basic_vec high = std::array<uint16_t, 16>{
        0xABCD, 0xABCD, 0xABCD, 0xABCD, 0xFFFF, 0x0000, 0xFFFF, 0x0000,
        0x0001, 0x0000, 0x5555, 0xAAAA, 0xDEAD, 0xCAFE, 0x8000, 0xBEEF};
    basic_vec low = std::array<uint16_t, 16>{
        0x1234, 0x1234, 0x1234, 0x1234, 0x0000, 0xFFFF, 0xFFFF, 0x0000,
        0x0000, 0x8000, 0xAAAA, 0x5555, 0xBEEF, 0xBABE, 0x0001, 0xBEEF};
    basic_vec shifts = std::array<uint16_t, 16>{
        0, 8, 15, 1, 8, 8, 5, 7, 1, 1, 4, 4, 12, 3, 15, 9};

    auto got = funnel_shift_right(high, low, shifts);
    BOOST_SIMD_EQUAL(got,
        applyTernary(high, low, shifts,
            [](uint16_t h, uint16_t l, uint16_t s) -> uint16_t { return ref::fsr(h, l, static_cast<unsigned>(s)); }));

    BOOST_CHECK_EQUAL(got[0], uint16_t(0x1234));  // shift==0 => low
    BOOST_CHECK_EQUAL(got[1], uint16_t(0xCD12));  // proposal diagram
    BOOST_CHECK_EQUAL(got[6], uint16_t(0xFFFF));  // all-ones
    BOOST_CHECK_EQUAL(got[7], uint16_t(0x0000));  // all-zero
}

BOOST_AUTO_TEST_CASE(left_mixed) {
    basic_vec high = std::array<uint16_t, 16>{
        0xABCD, 0xABCD, 0xABCD, 0xABCD, 0xFFFF, 0x0000, 0xFFFF, 0x0000,
        0x0001, 0x0000, 0x5555, 0xAAAA, 0xDEAD, 0xCAFE, 0x8000, 0xBEEF};
    basic_vec low = std::array<uint16_t, 16>{
        0x1234, 0x1234, 0x1234, 0x1234, 0x0000, 0xFFFF, 0xFFFF, 0x0000,
        0x0000, 0x8000, 0xAAAA, 0x5555, 0xBEEF, 0xBABE, 0x0001, 0xBEEF};
    basic_vec shifts = std::array<uint16_t, 16>{
        0, 8, 15, 1, 8, 8, 5, 7, 1, 1, 4, 4, 12, 3, 15, 9};

    auto got = funnel_shift_left(high, low, shifts);
    BOOST_SIMD_EQUAL(got,
        applyTernary(high, low, shifts,
            [](uint16_t h, uint16_t l, uint16_t s) -> uint16_t { return ref::fsl(h, l, static_cast<unsigned>(s)); }));

    BOOST_CHECK_EQUAL(got[0], uint16_t(0xABCD));  // shift==0 => high
    BOOST_CHECK_EQUAL(got[6], uint16_t(0xFFFF));  // all-ones
    BOOST_CHECK_EQUAL(got[7], uint16_t(0x0000));  // all-zero
}

BOOST_AUTO_TEST_SUITE_END()

// ===================================================================
// 10. CONSTEXPR VERIFICATION
// ===================================================================

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_SUITE(constexpr_checks)

BOOST_AUTO_TEST_CASE(constexpr_right_uniform) {
    constexpr auto high = vec<uint16_t, 4>(uint16_t(0xABCD));
    constexpr auto low  = vec<uint16_t, 4>(uint16_t(0x1234));
    constexpr auto got  = funnel_shift_right(high, low, 8);
    static_assert(got[0] == 0xCD12);
    static_assert(got[1] == 0xCD12);
    static_assert(got[2] == 0xCD12);
    static_assert(got[3] == 0xCD12);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(constexpr_left_uniform) {
    constexpr auto high = vec<uint16_t, 4>(uint16_t(0xABCD));
    constexpr auto low  = vec<uint16_t, 4>(uint16_t(0x1234));
    constexpr auto got  = funnel_shift_left(high, low, 8);
    constexpr auto exp  = funnel_shift_right(high, low, 8);
    static_assert(got[0] == exp[0]);
    static_assert(got[1] == exp[1]);
    static_assert(got[2] == exp[2]);
    static_assert(got[3] == exp[3]);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(constexpr_right_zero_returns_low) {
    constexpr auto high = vec<uint32_t, 2>(0xDEADBEEFu);
    constexpr auto low  = vec<uint32_t, 2>(0x12345678u);
    constexpr auto got  = funnel_shift_right(high, low, 0);
    static_assert(got[0] == 0x12345678u);
    static_assert(got[1] == 0x12345678u);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(constexpr_left_zero_returns_high) {
    constexpr auto high = vec<uint32_t, 2>(0xDEADBEEFu);
    constexpr auto low  = vec<uint32_t, 2>(0x12345678u);
    constexpr auto got  = funnel_shift_left(high, low, 0);
    static_assert(got[0] == 0xDEADBEEFu);
    static_assert(got[1] == 0xDEADBEEFu);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(constexpr_per_element) {
    constexpr auto high   = vec<uint8_t, 4>(uint8_t(0xFF));
    constexpr auto low    = vec<uint8_t, 4>(uint8_t(0x00));
    constexpr auto shifts = vec<uint8_t, 4>([](auto i) { return uint8_t(i); });
    constexpr auto got    = funnel_shift_right(high, low, shifts);
    static_assert(got[0] == 0x00);  // shift 0 => low
    static_assert(got[1] == 0x80);  // shift 1 => MSB from high slides in
    static_assert(got[2] == 0xC0);  // shift 2
    static_assert(got[3] == 0xE0);  // shift 3
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(constexpr_rotation) {
    constexpr auto v   = vec<uint16_t, 4>(uint16_t(0xBEEF));
    constexpr auto got = funnel_shift_right(v, v, 4);
    static_assert(got[0] == 0xFBEE);
    static_assert(got[1] == 0xFBEE);
    static_assert(got[2] == 0xFBEE);
    static_assert(got[3] == 0xFBEE);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()
#endif // defined(_XVEC_HAS_CONSTEXPR)
