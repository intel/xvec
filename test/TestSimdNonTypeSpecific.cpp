//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimdTestUtilities.hpp"

using xvec::simd::flags;

struct VectorizableStruct {
  int a, b;

  friend VectorizableStruct operator+(const VectorizableStruct& lhs, const VectorizableStruct& rhs) {
    return {lhs.a + rhs.a, lhs.b + rhs.b};
  }
  friend VectorizableStruct operator-(const VectorizableStruct& lhs, const VectorizableStruct& rhs) {
    return {lhs.a - rhs.a, lhs.b - rhs.b};
  }

  friend VectorizableStruct operator-(const VectorizableStruct& rhs) {
    return {-rhs.a, -rhs.b};
  }
};

struct NonVectorizableStruct {
  int a[10]; // Too big to fit in a vec, so not vectorizable.
};

// Has operator+ but it returns something other than InvalidOperatorStruct
struct InvalidOperatorStruct
{
  int a;
  friend float operator+(const InvalidOperatorStruct& lhs, const InvalidOperatorStruct& rhs) { return float(lhs.a + rhs.a); }
  friend float operator-(const InvalidOperatorStruct& rhs) { return float(-rhs.a); }
};

struct Empty {};
union Union { int a; float b; };

enum Colours {RED, GREEN, BLUE, YELLOW, ORANGE};

enum class StrictColours {PURPLE, CYAN, MAGENTA};

void CheckSize()
{
  // Make sure that selected sizes are exactly as expected. This ensures that no unwanted data is
  // creeping in.
  static_assert(sizeof(xvec::simd::vec<float, 4>) == 16);
  static_assert(sizeof(xvec::simd::vec<float, 8>) == 32);
  static_assert(sizeof(xvec::simd::vec<float, 16>) == 64);
  static_assert(sizeof(xvec::simd::vec<char, 16>) == 16);
  static_assert(sizeof(xvec::simd::vec<char, 32>) == 32);
  static_assert(sizeof(xvec::simd::vec<char, 64>) == 64);
}

void QueryFlags()
{
  // Query the presence of a flag
  using F = flags<xvec::simd::detail::flag_unchecked, xvec::simd::detail::flag_convert>;

  // Empty
  static_assert(!xvec::simd::detail::contains_flag<xvec::simd::detail::flag_unchecked>(xvec::simd::flag_default));

  // Contained flags
  static_assert(xvec::simd::detail::contains_flag<xvec::simd::detail::flag_unchecked>(F{}));
  static_assert(xvec::simd::detail::contains_flag<xvec::simd::detail::flag_convert>(F{}));

  // Non-contained flag
  static_assert(!xvec::simd::detail::contains_flag<xvec::simd::detail::flag_aligned>(F{}));
}

void CreateFlags()
{
  auto flags = xvec::simd::flag_default | xvec::simd::flag_convert;

  // Same checks as above, but on a constructed flag set.
  static_assert(xvec::simd::detail::contains_flag<xvec::simd::detail::flag_convert>(flags));
  static_assert(!xvec::simd::detail::contains_flag<xvec::simd::detail::flag_aligned>(flags));
}

void TestRankConversionCheck()
{
  static_assert(!xvec::simd::detail::higher_floating_point_rank<float, double>);
  static_assert(!xvec::simd::detail::higher_floating_point_rank<float, float>);
  static_assert(!xvec::simd::detail::higher_floating_point_rank<double, double>);

  static_assert(xvec::simd::detail::higher_floating_point_rank<double, float>);

#if defined(__FLT16_MIN__)
  static_assert(!xvec::simd::detail::higher_floating_point_rank<_Float16, float>);
  static_assert(!xvec::simd::detail::higher_floating_point_rank<_Float16, double>);

  static_assert(xvec::simd::detail::higher_floating_point_rank<float, _Float16>);
  static_assert(xvec::simd::detail::higher_floating_point_rank<double, _Float16>);
#endif

  static_assert(!xvec::simd::detail::higher_integer_rank<signed char, short>);
  static_assert(!xvec::simd::detail::higher_integer_rank<short, int>);
  static_assert(!xvec::simd::detail::higher_integer_rank<int, long>);
  static_assert(!xvec::simd::detail::higher_integer_rank<short, short>);
  static_assert(!xvec::simd::detail::higher_integer_rank<long, long>);

  static_assert(xvec::simd::detail::higher_integer_rank<short, char>);
  static_assert(xvec::simd::detail::higher_integer_rank<long, int>);
  static_assert(xvec::simd::detail::higher_integer_rank<long, signed char>);
}

void TestImplicit()
{
  static_assert(!xvec::simd::detail::needs_explicit_conversion<float, double>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<float, float>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<double, double>);

  static_assert(xvec::simd::detail::needs_explicit_conversion<double, float>);

#if defined(__FLT16_MIN__)
  static_assert(!xvec::simd::detail::needs_explicit_conversion<_Float16, float>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<_Float16, double>);

  static_assert(xvec::simd::detail::needs_explicit_conversion<float, _Float16>);
  static_assert(xvec::simd::detail::needs_explicit_conversion<double, _Float16>);
#endif

  static_assert(!xvec::simd::detail::needs_explicit_conversion<signed char, short>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<short, int>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<int, long>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<short, short>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<long, long>);

  static_assert(xvec::simd::detail::needs_explicit_conversion<short, char>);
  static_assert(xvec::simd::detail::needs_explicit_conversion<long, int>);
  static_assert(xvec::simd::detail::needs_explicit_conversion<long, signed char>);

  // Mixed types.
  static_assert(!xvec::simd::detail::needs_explicit_conversion<signed char, float>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<short, float>);
  static_assert(!xvec::simd::detail::needs_explicit_conversion<int, double>);

  static_assert(xvec::simd::detail::needs_explicit_conversion<float, short>);
  static_assert(xvec::simd::detail::needs_explicit_conversion<double, int>);
  static_assert(xvec::simd::detail::needs_explicit_conversion<double, long>);
}

void TestSimdSameSize()
{
  using xvec::simd::vec;
  using xvec::simd::mask;

  using V0 = vec<float, 10>;
  using V1 = vec<float, 4>;
  using V2 = vec<int, 10>;
  static_assert(xvec::simd::simd_same_size<V0, V0>);      // Same as self
  static_assert(xvec::simd::simd_same_size<V0, V2>);      // Different type, same size
  static_assert(!xvec::simd::simd_same_size<V0, V1>);     // Different sizes
  static_assert(!xvec::simd::simd_same_size<V1, V2>);     // Different sizes

  using M0 = mask<float, 10>;
  using M1 = mask<float, 4>;
  using M2 = mask<int, 10>;
  static_assert(xvec::simd::simd_same_size<M0, M0>);      // Same as self
  static_assert(xvec::simd::simd_same_size<M0, M2>);      // Different type, same size
  static_assert(!xvec::simd::simd_same_size<M0, M1>);     // Different sizes
  static_assert(!xvec::simd::simd_same_size<M1, M2>);     // Different sizes

  // Mixture of vec/mask
  static_assert(xvec::simd::simd_same_size<V0, M0>);
  static_assert(xvec::simd::simd_same_size<M0, V2>);
  static_assert(!xvec::simd::simd_same_size<V0, V1>);     // Different sizes
  static_assert(!xvec::simd::simd_same_size<V1, V2>);     // Different sizes

  static_assert(!xvec::simd::simd_same_size<int, float>); // Not simd at all

}

void TestBuiltinVectorizable()
{
  static_assert(xvec::simd::detail::builtin_vectorizable<unsigned char>);
  static_assert(xvec::simd::detail::builtin_vectorizable<signed char>);
  static_assert(xvec::simd::detail::builtin_vectorizable<char>);
  static_assert(xvec::simd::detail::builtin_vectorizable<short>);
  static_assert(xvec::simd::detail::builtin_vectorizable<unsigned short>);
  static_assert(xvec::simd::detail::builtin_vectorizable<int>);
  static_assert(xvec::simd::detail::builtin_vectorizable<unsigned int>);
  static_assert(xvec::simd::detail::builtin_vectorizable<long>);
  static_assert(xvec::simd::detail::builtin_vectorizable<unsigned long>);
  static_assert(xvec::simd::detail::builtin_vectorizable<long long>);
  static_assert(xvec::simd::detail::builtin_vectorizable<unsigned long long>);
  static_assert(xvec::simd::detail::builtin_vectorizable<float>);
  static_assert(xvec::simd::detail::builtin_vectorizable<double>);

  static_assert(!xvec::simd::detail::builtin_vectorizable<long double>);

  static_assert(xvec::simd::detail::builtin_vectorizable<std::complex<float>>);
  static_assert(xvec::simd::detail::builtin_vectorizable<std::complex<double>>);

  static_assert(!xvec::simd::detail::builtin_vectorizable<std::complex<long double>>);

  static_assert(xvec::simd::detail::builtin_vectorizable<std::byte>);

#ifdef __FLT16_MIN__
  static_assert(xvec::simd::detail::builtin_vectorizable<_Float16>);
  static_assert(xvec::simd::detail::builtin_vectorizable<std::complex<_Float16>>);
#endif

  // Extended types.
  static_assert(xvec::simd::detail::builtin_vectorizable<unsigned __int128>);
  static_assert(xvec::simd::detail::builtin_vectorizable<signed __int128>);
  static_assert(!xvec::simd::detail::builtin_vectorizable<Colours>);

  static_assert(!xvec::simd::detail::builtin_vectorizable<bool>);

  static_assert(!xvec::simd::detail::builtin_vectorizable<VectorizableStruct>);
  static_assert(!xvec::simd::detail::builtin_vectorizable<std::string>);
  static_assert(!xvec::simd::detail::builtin_vectorizable<std::list<int>>);
  static_assert(!xvec::simd::detail::builtin_vectorizable<std::vector<int>>);

  static_assert(!xvec::simd::detail::builtin_vectorizable<xvec::simd::vec<float>>);
}

void TestUdtVectorizable()
{
  static_assert(!xvec::simd::detail::udt_vectorizable<unsigned char>);
  static_assert(!xvec::simd::detail::udt_vectorizable<signed char>);
  static_assert(!xvec::simd::detail::udt_vectorizable<char>);
  static_assert(!xvec::simd::detail::udt_vectorizable<short>);
  static_assert(!xvec::simd::detail::udt_vectorizable<unsigned short>);
  static_assert(!xvec::simd::detail::udt_vectorizable<int>);
  static_assert(!xvec::simd::detail::udt_vectorizable<unsigned int>);
  static_assert(!xvec::simd::detail::udt_vectorizable<long>);
  static_assert(!xvec::simd::detail::udt_vectorizable<unsigned long>);
  static_assert(!xvec::simd::detail::udt_vectorizable<long long>);
  static_assert(!xvec::simd::detail::udt_vectorizable<unsigned long long>);
  static_assert(!xvec::simd::detail::udt_vectorizable<float>);
  static_assert(!xvec::simd::detail::udt_vectorizable<double>);

  static_assert(!xvec::simd::detail::udt_vectorizable<long double>);

  static_assert(!xvec::simd::detail::udt_vectorizable<std::complex<float>>);
  static_assert(!xvec::simd::detail::udt_vectorizable<std::complex<double>>);

  static_assert(!xvec::simd::detail::udt_vectorizable<std::complex<long double>>);

#ifdef __FLT16_MIN__
  static_assert(!xvec::simd::detail::udt_vectorizable<_Float16>);
  static_assert(!xvec::simd::detail::udt_vectorizable<std::complex<_Float16>>);
#endif

  // Extended types.
  static_assert(!xvec::simd::detail::udt_vectorizable<unsigned __int128>);
  static_assert(!xvec::simd::detail::udt_vectorizable<signed __int128>);
  static_assert(!xvec::simd::detail::udt_vectorizable<std::byte>);
  static_assert(xvec::simd::detail::udt_vectorizable<Colours>);
  static_assert(xvec::simd::detail::udt_vectorizable<StrictColours>);

  static_assert(xvec::simd::detail::udt_vectorizable<bool>);

  static_assert(xvec::simd::detail::udt_vectorizable<VectorizableStruct>);
  static_assert(!xvec::simd::detail::udt_vectorizable<NonVectorizableStruct>);
  static_assert(!xvec::simd::detail::udt_vectorizable<std::string>);
  static_assert(!xvec::simd::detail::udt_vectorizable<std::list<int>>);
  static_assert(!xvec::simd::detail::udt_vectorizable<std::vector<int>>);

  static_assert(!xvec::simd::detail::udt_vectorizable<xvec::simd::vec<float>>);

}

void TestVectorizable()
{
  static_assert(xvec::simd::detail::vectorizable<unsigned char>);
  static_assert(xvec::simd::detail::vectorizable<signed char>);
  static_assert(xvec::simd::detail::vectorizable<char>);
  static_assert(xvec::simd::detail::vectorizable<short>);
  static_assert(xvec::simd::detail::vectorizable<unsigned short>);
  static_assert(xvec::simd::detail::vectorizable<int>);
  static_assert(xvec::simd::detail::vectorizable<unsigned int>);
  static_assert(xvec::simd::detail::vectorizable<long>);
  static_assert(xvec::simd::detail::vectorizable<unsigned long>);
  static_assert(xvec::simd::detail::vectorizable<long long>);
  static_assert(xvec::simd::detail::vectorizable<unsigned long long>);
  static_assert(xvec::simd::detail::vectorizable<float>);
  static_assert(xvec::simd::detail::vectorizable<double>);

  static_assert(!xvec::simd::detail::vectorizable<long double>);

  static_assert(xvec::simd::detail::vectorizable<std::complex<float>>);
  static_assert(xvec::simd::detail::vectorizable<std::complex<double>>);
  static_assert(!xvec::simd::detail::vectorizable<std::complex<long double>>);

#ifdef __FLT16_MIN__
  static_assert(xvec::simd::detail::vectorizable<_Float16>);
  static_assert(xvec::simd::detail::vectorizable<std::complex<_Float16>>);
#endif

  // Extended types.
  static_assert(xvec::simd::detail::vectorizable<unsigned __int128>);
  static_assert(xvec::simd::detail::vectorizable<signed __int128>);
  static_assert(xvec::simd::detail::vectorizable<Colours>);
  static_assert(xvec::simd::detail::vectorizable<StrictColours>);
  static_assert(xvec::simd::detail::vectorizable<std::byte>);

  static_assert(xvec::simd::detail::vectorizable<bool>);

  static_assert(xvec::simd::detail::vectorizable<VectorizableStruct>);

  static_assert(!xvec::simd::detail::vectorizable<std::string>);
  static_assert(!xvec::simd::detail::vectorizable<std::list<int>>);
  static_assert(!xvec::simd::detail::vectorizable<std::vector<int>>);

  static_assert(!xvec::simd::detail::vectorizable<int*>);
  static_assert(!xvec::simd::detail::vectorizable<int VectorizableStruct::*>);
  static_assert(!xvec::simd::detail::vectorizable<const int>);
  static_assert(!xvec::simd::detail::vectorizable<Empty>);
  static_assert(!xvec::simd::detail::vectorizable<Union>);

  static_assert(!xvec::simd::detail::vectorizable<xvec::simd::vec<float>>);
}

void TestConcepts()
{
  using xvec::simd::vec;
  using xvec::simd::mask;


  using sf = vec<float>;
  using shf = vec<_Float16>;
  using sint = vec<int>;
  using suint = vec<unsigned>;
  using scmplx = vec<std::complex<float>>;
  using scolours = vec<Colours>;

  using smf = mask<float>;
  using smi = mask<int>;

  // simd types.
  static_assert(xvec::simd::vec_type<sf>);
  static_assert(xvec::simd::vec_type<shf>);
  static_assert(xvec::simd::vec_type<sint>);
  static_assert(xvec::simd::vec_type<suint>);
  static_assert(xvec::simd::vec_type<scolours>);
  static_assert(!xvec::simd::vec_type<smf>);
  static_assert(!xvec::simd::vec_type<smi>);

  // simd mask types.
  static_assert(!xvec::simd::mask_type<sf>);
  static_assert(!xvec::simd::mask_type<shf>);
  static_assert(!xvec::simd::mask_type<sint>);
  static_assert(!xvec::simd::mask_type<suint>);
  static_assert(!xvec::simd::mask_type<scolours>);
  static_assert(xvec::simd::mask_type<smf>);
  static_assert(xvec::simd::mask_type<smi>);

  // Float
  static_assert(xvec::simd::vec_floating_point<sf>);
  static_assert(xvec::simd::vec_floating_point<shf>);
  static_assert(!xvec::simd::vec_floating_point<sint>);
  static_assert(!xvec::simd::vec_floating_point<suint>);
  static_assert(!xvec::simd::vec_floating_point<scmplx>);
  static_assert(!xvec::simd::vec_floating_point<scolours>);

  // Integral
  static_assert(!xvec::simd::vec_integral<sf>);
  static_assert(!xvec::simd::vec_integral<shf>);
  static_assert(xvec::simd::vec_integral<sint>);
  static_assert(xvec::simd::vec_integral<suint>);
  static_assert(!xvec::simd::vec_integral<scmplx>);
  static_assert(!xvec::simd::vec_integral<scolours>);

  // Signed
  static_assert(!xvec::simd::vec_signed_integral<sf>);
  static_assert(!xvec::simd::vec_signed_integral<shf>);
  static_assert(xvec::simd::vec_signed_integral<sint>);
  static_assert(!xvec::simd::vec_signed_integral<suint>);
  static_assert(!xvec::simd::vec_signed_integral<scmplx>);
  static_assert(!xvec::simd::vec_signed_integral<scolours>);

  // Unsigned integral
  static_assert(!xvec::simd::vec_unsigned_integral<sf>);
  static_assert(!xvec::simd::vec_unsigned_integral<shf>);
  static_assert(!xvec::simd::vec_unsigned_integral<sint>);
  static_assert(xvec::simd::vec_unsigned_integral<suint>);
  static_assert(!xvec::simd::vec_unsigned_integral<scmplx>);
  static_assert(!xvec::simd::vec_unsigned_integral<scolours>);

  // Unsigned integer (specifically fundamental integers, not integrals)
  static_assert(!xvec::simd::vec_unsigned_integer<sf>);
  static_assert(!xvec::simd::vec_unsigned_integer<shf>);
  static_assert(!xvec::simd::vec_unsigned_integer<sint>);
  static_assert(xvec::simd::vec_unsigned_integer<suint>);
  static_assert(!xvec::simd::vec_unsigned_integer<scmplx>);
  static_assert(!xvec::simd::vec_unsigned_integer<scolours>);

  // Arithmetic
  static_assert(xvec::simd::vec_arithmetic<sf>);
  static_assert(xvec::simd::vec_arithmetic<shf>);
  static_assert(xvec::simd::vec_arithmetic<sint>);
  static_assert(xvec::simd::vec_arithmetic<suint>);
  static_assert(!xvec::simd::vec_arithmetic<scmplx>);
  static_assert(!xvec::simd::vec_arithmetic<scolours>);

  // Complex
  static_assert(!xvec::simd::vec_complex<sf>);
  static_assert(!xvec::simd::vec_complex<shf>);
  static_assert(!xvec::simd::vec_complex<sint>);
  static_assert(!xvec::simd::vec_complex<suint>);
  static_assert(xvec::simd::vec_complex<scmplx>);
  static_assert(!xvec::simd::vec_complex<scolours>);

  // Totally ordered
  static_assert(xvec::simd::vec_totally_ordered<sf>);
  static_assert(xvec::simd::vec_totally_ordered<shf>);
  static_assert(xvec::simd::vec_totally_ordered<sint>);
  static_assert(xvec::simd::vec_totally_ordered<suint>);
  static_assert(!xvec::simd::vec_totally_ordered<scmplx>);
  static_assert(xvec::simd::vec_totally_ordered<scolours>);
}

void CheckMaskElementSize()
{
  static_assert(1 == xvec::simd::simd_mask_element_size_v<xvec::simd::mask<uint8_t>>);
  static_assert(2 == xvec::simd::simd_mask_element_size_v<xvec::simd::mask<uint16_t>>);
  static_assert(4 == xvec::simd::simd_mask_element_size_v<xvec::simd::mask<uint32_t>>);
  static_assert(8 == xvec::simd::simd_mask_element_size_v<xvec::simd::mask<uint64_t>>);
  static_assert(4 == xvec::simd::simd_mask_element_size_v<xvec::simd::mask<float>>);
}

void CheckCompatibleElements()
{
  using xvec::simd::vec_or_mask_same_element_type;

  static_assert(vec_or_mask_same_element_type<xvec::simd::vec<float, 8>, xvec::simd::vec<float, 4>>);
  static_assert(vec_or_mask_same_element_type<xvec::simd::mask<float, 8>, xvec::simd::mask<int32_t, 4>>);

  static_assert(!vec_or_mask_same_element_type<xvec::simd::vec<float, 8>, xvec::simd::vec<int32_t, 4>>);
  static_assert(!vec_or_mask_same_element_type<xvec::simd::mask<float, 8>, xvec::simd::mask<int16_t, 4>>);
}

void CheckPromotableType()
{
  using xvec::simd::detail::promotable_type;

  static_assert(promotable_type<uint8_t>);
  static_assert(promotable_type<short>);
  static_assert(promotable_type<int32_t>);
  static_assert(promotable_type<float>);
  static_assert(promotable_type<double>);
  static_assert(promotable_type<Colours>);
  static_assert(!promotable_type<StrictColours>);
  static_assert(!promotable_type<std::string>);
}

void CheckSupportedUnaryOp() {
  using xvec::simd::detail::supported_unary_op;

  static_assert(supported_unary_op<int8_t, std::negate<>>);
  static_assert(supported_unary_op<uint16_t, std::negate<>>);
  static_assert(supported_unary_op<int32_t, std::negate<>>);
  static_assert(supported_unary_op<uint64_t, std::negate<>>);
  static_assert(supported_unary_op<float, std::negate<>>);
  static_assert(supported_unary_op<double, std::negate<>>);

  static_assert(supported_unary_op<Colours, std::negate<>>);
  static_assert(!supported_unary_op<StrictColours, std::negate<>>);

  static_assert(supported_unary_op<VectorizableStruct, std::negate<>>);

  // It has negate, but it returns the wrong type.
  static_assert(!supported_unary_op<InvalidOperatorStruct, std::negate<>>);

  static_assert(!supported_unary_op<std::string, std::negate<>>);
}

void CheckSupportedBinaryOp() {
  using xvec::simd::detail::supported_binary_op;

  static_assert(supported_binary_op<int8_t, std::plus<>>);
  static_assert(supported_binary_op<uint16_t, std::minus<>>);
  static_assert(supported_binary_op<int32_t, std::multiplies<>>);
  static_assert(supported_binary_op<uint64_t, std::divides<>>);
  static_assert(supported_binary_op<float, std::minus<>>);
  static_assert(supported_binary_op<double, std::multiplies<>>);

  static_assert(supported_binary_op<Colours, std::plus<>>);
  static_assert(supported_binary_op<Colours, std::minus<>>);
  static_assert(!supported_binary_op<StrictColours, std::plus<>>);
  static_assert(!supported_binary_op<StrictColours, std::minus<>>);

  static_assert(supported_binary_op<VectorizableStruct, std::plus<>>);
  static_assert(supported_binary_op<VectorizableStruct, std::minus<>>);
  static_assert(!supported_binary_op<VectorizableStruct, std::multiplies<>>);
  static_assert(!supported_binary_op<VectorizableStruct, std::divides<>>);

  // It has +, but it returns the wrong type.
  static_assert(!supported_binary_op<InvalidOperatorStruct, std::plus<>>);

  static_assert(!supported_binary_op<std::string, std::divides<>>);
}

/// Check that value_preservingness_convertible_to is detectable.
void CheckValuePreserving()
{
  using xvec::simd::detail::value_preserving_convertible_to;

  static_assert(value_preserving_convertible_to<int8_t,  int16_t>);
  static_assert(value_preserving_convertible_to<uint8_t, int16_t>);
  static_assert(value_preserving_convertible_to<uint16_t, uint32_t>);
  static_assert(value_preserving_convertible_to<uint32_t, uint64_t>);
  static_assert(value_preserving_convertible_to<int32_t, double>);
  static_assert(value_preserving_convertible_to<uint32_t, double>);
#ifdef __FLT16_MIN__
  static_assert(value_preserving_convertible_to<_Float16, float>);
  static_assert(value_preserving_convertible_to<_Float16, double>);
#endif
  static_assert(value_preserving_convertible_to<float, double>);
  static_assert(value_preserving_convertible_to<float, long double>);
  static_assert(value_preserving_convertible_to<bool, float>);
  static_assert(value_preserving_convertible_to<bool, uint32_t>);
  static_assert(value_preserving_convertible_to<bool, bool>);

  // Integer downcast (fails)
  static_assert(!value_preserving_convertible_to<int32_t, int16_t>);
  // Unsigned int to signed int (fails if value > INT32_MAX)
  static_assert(!value_preserving_convertible_to<uint32_t, int32_t>);
  //  Signed int to unsigned int (fails for negatives)
  static_assert(!value_preserving_convertible_to<int32_t, uint32_t>);

  // Integer to floating point (fails for too-large ints)
  static_assert(!value_preserving_convertible_to<int64_t, float>);
  static_assert(!value_preserving_convertible_to<uint64_t, double>);
#ifdef __FLT16_MIN__
  static_assert(!value_preserving_convertible_to<int32_t, _Float16>);
  static_assert(!value_preserving_convertible_to<uint32_t, _Float16>);
#endif

  // Floating to integer (fails always except trivial cases)
  static_assert(!value_preserving_convertible_to<double, int32_t>);
  static_assert(!value_preserving_convertible_to<float, uint32_t>);

  // Floating to smaller floating (fails for overflow/precision loss)
  static_assert(!value_preserving_convertible_to<double, float>);
#ifdef __FLT16_MIN__
  static_assert(!value_preserving_convertible_to<float, _Float16>);
#endif

  // Anything to bool (except 0,1) fails.
  static_assert(!value_preserving_convertible_to<int32_t, bool>);
  static_assert(!value_preserving_convertible_to<double, bool>);
}

void GeneratorCheckLimits()
{
  struct test_generator {
    constexpr int operator()(int idx) const {
      return static_cast<int>(idx * 2);
    }
  };

  using xvec::simd::detail::generator_invocable;

  // Older compilers could't deal with large generator lists. Check that several different sizes work.
  static_assert(generator_invocable<test_generator, int, 8>);
  static_assert(generator_invocable<test_generator, int, 256>);
  static_assert(generator_invocable<test_generator, int, 512>);
  static_assert(generator_invocable<test_generator, int, 1024>);
}

void CheckRoundTripValuePreserved()
{
  using xvec::simd::detail::roundtrip_preserves_value;

  // exact int -> float round-trips (at boundary where float is exact for integers)
  static_assert( roundtrip_preserves_value<float>(16'777'216LL));  // 2^24 exact in float
  static_assert(!roundtrip_preserves_value<float>(16'777'217LL));  // 2^24+1 not exact in float

  // float -> int: integral-valued floats succeed, fractional fail
  static_assert( roundtrip_preserves_value<int>(2.0));
  static_assert(!roundtrip_preserves_value<int>(1.5));

  // float -> float: float value is representable in double and back (always exact in this direction)
  static_assert( roundtrip_preserves_value<double>(1.0f));

  // double -> float: many values *aren't* exactly representable
  // 0.1 is not exactly representable as binary float, so double(0.1f) -> float is exact,
  // but double literal 0.1 -> float -> double is not exact.
  static_assert( roundtrip_preserves_value<float>(0.1f));   // From is float here; round-trip to double and back is exact
  static_assert(!roundtrip_preserves_value<float>(0.1));     // From is double here; narrowing to float loses info
}

// Simple wrappers
struct W_int_200      { static constexpr int value = 200; };
struct W_int_300      { static constexpr int value = 300; };
struct W_int_neg1     { static constexpr int value = -1; };
struct W_u32_big      { static constexpr std::uint32_t value = 4'000'000'000u; };
struct W_i64_2p24_p1   { static constexpr std::int64_t value = 16'777'217; }; // 2^24 + 1
struct W_i64_2p24      { static constexpr std::int64_t value = 16'777'216; }; // 2^24
struct W_d_1_5        { static constexpr double value = 1.5; };
struct W_d_2_0        { static constexpr double value = 2.0; };
struct W_f_0_1        { static constexpr float value = 0.1f; };
struct W_f_1_0        { static constexpr float value = 1.0f; };

void CheckValueIsRepresentableBy()
{
  using xvec::simd::detail::wrapped_value_is_representable_by;

  // int wrappers to uint8_t
  static_assert( wrapped_value_is_representable_by<W_int_200, std::uint8_t>);
  static_assert(!wrapped_value_is_representable_by<W_int_300, std::uint8_t>);
  static_assert(!wrapped_value_is_representable_by<W_int_neg1, std::uint8_t>);

  // uint32_t big to int32_t (doesn't fit)
  static_assert(!wrapped_value_is_representable_by<W_u32_big, std::int32_t>);

  // int64 -> float: precision boundary
  static_assert( wrapped_value_is_representable_by<W_i64_2p24, float>);
  static_assert(!wrapped_value_is_representable_by<W_i64_2p24_p1, float>);

  // double -> int: exact integral double ok, fractional not ok
  static_assert( wrapped_value_is_representable_by<W_d_2_0, int>);
  static_assert(!wrapped_value_is_representable_by<W_d_1_5, int>);
}

struct NoConvert {
  static constexpr int value = 1;
};

struct NoDefault {
  static constexpr int value = 3;
  NoDefault() = delete;
  constexpr operator int() const { return value; }
  friend constexpr bool operator==(NoDefault, int x) { return value == x; }
  friend constexpr bool operator==(int x, NoDefault) { return x == value; }
};

void CheckConstantWrapperLike()
{
  using xvec::simd::detail::constexpr_wrapper_like;

  // integral_constant should always work
  static_assert(constexpr_wrapper_like<std::integral_constant<int, 200>>);
  static_assert(constexpr_wrapper_like<std::integral_constant<int, 0>>);
  static_assert(constexpr_wrapper_like<std::integral_constant<int, 200>>);
  static_assert(constexpr_wrapper_like<std::true_type>);
  static_assert(constexpr_wrapper_like<std::false_type>);

  // Should FAIL: has ::value but not convertible to underlying type
  static_assert(!constexpr_wrapper_like<NoConvert>);

  // Should FAIL: not default-constructible (T{} ill-formed)
  static_assert(!constexpr_wrapper_like<NoDefault>);
}

struct ExplicitInt {
  explicit ExplicitInt(int) {}
};

struct NotConstructible {
  NotConstructible() = default;
  NotConstructible(int) = delete;
};

// Non-arithmetic, not std::integral_constant, but convertible to int
struct NonArithToInt {
  constexpr operator int() const { return 7; }
};

// Not convertible to int
struct NonArithNotConvertible {};

void CheckBroadcastConstructible()
{
  using xvec::simd::detail::broadcast_constructible;

  // 0) Hard requirement: std::convertible_to<From, value_type>
  static_assert(!broadcast_constructible<NonArithNotConvertible, std::int32_t>,
                "Must be false if std::convertible_to<From, value_type> is not satisfied.");

  // 1) Clause 1: From is not arithmetic and not wrapper-like => only needs convertible_to
  static_assert(broadcast_constructible<NonArithToInt, std::int32_t>,
                "Non-arithmetic, non-wrapper-like, convertible => should satisfy.");

  // 2) Clause 2: arithmetic and value-preserving conversion
  static_assert(broadcast_constructible<std::int32_t, std::int64_t>,
                "int32->int64 should be value-preserving => true.");
  static_assert(broadcast_constructible<std::uint8_t, std::int32_t>,
                "uint8->int32 should be value-preserving => true.");

  // All int32 values are exactly representable in double
  static_assert(broadcast_constructible<std::int32_t, double>,
                "int32->double is typically value-preserving (exact) => true.");

  // Commonly NOT value-preserving
  static_assert(!broadcast_constructible<std::int64_t, float>,
                "int64->float is typically NOT value-preserving => false.");
  static_assert(!broadcast_constructible<double, float>,
                "double->float is typically NOT value-preserving => false.");
  static_assert(!broadcast_constructible<std::int32_t, std::int8_t>,
                "int32->int8 is narrowing, typically NOT value-preserving => false.");

  static_assert(broadcast_constructible<bool, int>,
                "bool->int commonly considered value-preserving (0/1) => true.");
  static_assert(!broadcast_constructible<int, bool>,
                "int->bool is not value-preserving => false.");

#if defined(__FLT16_MIN__)
  static_assert(broadcast_constructible<std::integral_constant<int, 0>, _Float16>,
                "0 is representable by fp16 => true.");
  static_assert(broadcast_constructible<std::integral_constant<int, 1>, _Float16>,
                "1 is representable by fp16 => true.");
  static_assert(broadcast_constructible<std::integral_constant<int, -1>, _Float16>,
                "-1 is representable by fp16 => true.");

  // IEEE-754 binary16: all integers in [-2048, 2048] are exactly representable.
  static_assert(broadcast_constructible<std::integral_constant<int, 2048>, _Float16>,
                "2048 is exactly representable by fp16 => true.");
  static_assert(!broadcast_constructible<std::integral_constant<int, 2049>, _Float16>,
                "2049 is not exactly representable by fp16 => false.");

  // binary16 max finite is 65504
  static_assert(broadcast_constructible<std::integral_constant<int, 65504>, _Float16>,
                "65504 (max finite fp16) is representable by fp16 => true.");
  static_assert(!broadcast_constructible<std::integral_constant<int, 65505>, _Float16>,
                "65505 is not representable as finite fp16 => false.");

  static_assert(!broadcast_constructible<float, _Float16>,
                "float->fp16 is typically NOT value-preserving => false.");
  static_assert(!broadcast_constructible<double, _Float16>,
                "double->fp16 is typically NOT value-preserving => false.");
#endif

  // 3) Clause 3: wrapper-like (std::integral_constant), arithmetic underlying,
  // and constant representable by value_type
  static_assert(broadcast_constructible<std::integral_constant<int, 0>, std::uint8_t>,
                "0 representable by uint8 => true.");
  static_assert(broadcast_constructible<std::integral_constant<int, 255>, std::uint8_t>,
                "255 representable by uint8 => true.");
  static_assert(!broadcast_constructible<std::integral_constant<int, 256>, std::uint8_t>,
                "256 not representable by uint8 => false.");
  static_assert(!broadcast_constructible<std::integral_constant<int, -1>, std::uint8_t>,
                "-1 not representable by uint8 => false.");

  static_assert(broadcast_constructible<std::integral_constant<int, -128>, std::int8_t>,
                "-128 representable by int8 => true.");
  static_assert(broadcast_constructible<std::integral_constant<int, 127>, std::int8_t>,
                "127 representable by int8 => true.");
  static_assert(!broadcast_constructible<std::integral_constant<int, 128>, std::int8_t>,
                "128 not representable by int8 => false.");
  static_assert(!broadcast_constructible<std::integral_constant<int, -129>, std::int8_t>,
                "-129 not representable by int8 => false.");

  // Unsigned boundary: 2^63 is not representable by int64
  static_assert(
    !broadcast_constructible<
      std::integral_constant<std::uint64_t, (std::uint64_t{1} << 63)>,
      std::int64_t
    >,
    "2^63 not representable by int64 => false."
  );

  // 3) Floating representability tests
  // 16777217 (2^24 + 1) is not exactly representable in float (mantissa = 24 bits incl hidden bit)
  static_assert(!broadcast_constructible<std::integral_constant<int, 16'777'217>, float>,
                "16777217 not exactly representable as float => false.");
  static_assert(broadcast_constructible<std::integral_constant<int, 16'777'217>, double>,
                "16777217 exactly representable as double => true.");

  // 3) Ensure wrapper-like path still requires std::convertible_to<From, value_type>.
  // integral_constant<int,7> is convertible to int, so this should be OK:
  static_assert(broadcast_constructible<std::integral_constant<int, 7>, int>,
                "integral_constant<int,7> convertible to int and representable => true.");

  // But if value_type is unrelated, convertible_to should fail and concept must be false:
  struct NotConvertibleFromInt {};
  static_assert(!broadcast_constructible<std::integral_constant<int, 7>, NotConvertibleFromInt>,
                "Must fail convertible_to requirement, even for wrapper-like constants.");

  // 3) Ensure wrapper-like constant does NOT bypass representability:
  static_assert(!broadcast_constructible<std::integral_constant<int, 300>, std::uint8_t>,
                "Wrapper-like: if constant isn't representable, must be false (no fallback).");

  // Scalar to complex conversions.
  static_assert( broadcast_constructible<float, std::complex<float>>);
  static_assert( broadcast_constructible<float, std::complex<double>>);
  static_assert(!broadcast_constructible<double, std::complex<float>>);
}

BOOST_AUTO_TEST_CASE(NonTypeSpecific){}  // Make sure we have something in our test tree!
