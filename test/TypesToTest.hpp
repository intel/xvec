//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

 #pragma once

#include <utility>
#include <iostream> // Must appear before <xvec/simd> to enable operator output.
#include <xvec/simd>
#include <boost/mpl/list.hpp>
#include <boost/mp11.hpp>
#include <boost/test/unit_test.hpp>
#include <random>
#include <type_traits>
#include <concepts>

#include "EnumTestTypes.hpp"
#include "UserDefinedTestTypes.hpp"

namespace mp11 = boost::mp11;

// Helper to conditionally define type lists
template<bool Condition, typename... Types>
using type_list_if = std::conditional_t<Condition, mp11::mp_list<Types...>, mp11::mp_list<>>;

// Basic numeric types
using SignedIntegerTypes = mp11::mp_list<int8_t, int16_t, int32_t, int64_t>;
using UnsignedIntegerTypes = mp11::mp_list<uint8_t, uint16_t, uint32_t, uint64_t>;
using UniversalFloatTypes = mp11::mp_list<float, double>;
using UniversalComplexTypes = mp11::mp_list<std::complex<float>>;

// Conditional types based on compiler support
#if defined(__FLT16_MIN__)
  using Fp16FloatTypes = mp11::mp_list<_Float16>;
  using Fp16ComplexTypes = mp11::mp_list<std::complex<_Float16>>;
#else
  using Fp16FloatTypes = mp11::mp_list<>;
  using Fp16ComplexTypes = mp11::mp_list<>;
#endif

#if defined(__clang__)
  using WideComplexTypes = mp11::mp_list<std::complex<double>>;
#else
  using WideComplexTypes = mp11::mp_list<>;
#endif

// Aggregate float and complex types
using FloatTypes = mp11::mp_append<Fp16FloatTypes, UniversalFloatTypes>;
using ComplexTypes = mp11::mp_append<Fp16ComplexTypes, UniversalComplexTypes, WideComplexTypes>;

// Enum and user-defined types
using UnscopedEnumTypes = mp11::mp_list<UnscopedEnumTest>;
using ScopedEnumTypes = mp11::mp_list<ScopedEnumTest, TypedEnumTest>;
using EnumTypes = mp11::mp_append<UnscopedEnumTypes, ScopedEnumTypes>;
using UserTypes = mp11::mp_list<Meters, UserDefinedInteger>;

// Test sizes
#if defined(_XVEC_TEST_FAST)
  using AllSizes = mp11::mp_list_c<int, 8, 15, 64>;
#else
  using AllSizes = mp11::mp_list_c<int, 1, 2, 3, 4, 7, 8, 16, 29, 32, 64, 99>;
#endif

// Template helpers
template<typename T, typename CN> using to_vec = xvec::simd::vec<T, CN::value>;
template<typename T, typename CN> using to_mask = xvec::simd::mask<T, CN::value>;
template<typename T> using to_native_vec = xvec::simd::vec<T>;

// All base types
using AllTypes = mp11::mp_append<SignedIntegerTypes, UnsignedIntegerTypes, 
                                  FloatTypes, ComplexTypes, EnumTypes, UserTypes>;
using IntegerTypes = mp11::mp_append<SignedIntegerTypes, UnsignedIntegerTypes>;
using ArithmeticTypes = mp11::mp_append<IntegerTypes, FloatTypes, mp11::mp_list<Meters>>;

// Helper to generate SIMD types (native or all sizes)
#if defined(_XVEC_TEST_NATIVE)
  template<typename TypeList> using to_simd = mp11::mp_transform<xvec::simd::vec, TypeList>;
#else
  template<typename TypeList> using to_simd = mp11::mp_product<to_vec, TypeList, AllSizes>;
#endif

// SIMD type lists
using AllSimdTypes = to_simd<AllTypes>;
using FloatSimdTypes = to_simd<FloatTypes>;
using ComplexSimdTypes = to_simd<ComplexTypes>;
using UnsignedSimdTypes = to_simd<UnsignedIntegerTypes>;
using IntegerSimdTypes = to_simd<IntegerTypes>;
using UserSimdTypes = to_simd<UserTypes>;
using ByteSimdTypes = to_simd<mp11::mp_list<std::byte>>;
using ScopedEnumSimdTypes = mp11::mp_product<to_vec, ScopedEnumTypes, AllSizes>;
using UnscopedEnumSimdTypes = mp11::mp_product<to_vec, UnscopedEnumTypes, AllSizes>;
using EnumSimdTypes = mp11::mp_append<UnscopedEnumSimdTypes, ScopedEnumSimdTypes>;

// Mask types (unique by size)
using AllSimdMaskTypes = mp11::mp_unique<mp11::mp_product<to_mask, AllTypes, AllSizes>>;

// Filtered SIMD types using concepts
template<typename T> 
using is_arithmetic = mp11::mp_bool<xvec::simd::detail::is_floating_point<typename T::value_type>() 
                                     || std::integral<typename T::value_type>>;

template<typename T> 
using is_comparable = mp11::mp_bool<requires(typename T::value_type x) { x < x; }>;

template<typename T> 
using is_bitmaskable = mp11::mp_bool<requires(typename T::value_type x) { x & x; }>;

template<typename T> using is_non_scalar = mp11::mp_bool<(T::size() > 1)>;

using ArithmeticSimdTypes = mp11::mp_copy_if<AllSimdTypes, is_arithmetic>;
using OrderableSimdTypes = mp11::mp_copy_if<AllSimdTypes, is_comparable>;
using BitmaskSimdTypes = mp11::mp_copy_if<AllSimdTypes, is_bitmaskable>;

// Specialized type groups
using NumericSimdTypes = mp11::mp_append<IntegerSimdTypes, FloatSimdTypes, ComplexSimdTypes, 
                                          UnscopedEnumSimdTypes, UserSimdTypes>;
using PermuteTestTypes = 
  mp11::mp_copy_if<mp11::mp_append<UnsignedSimdTypes, ComplexSimdTypes, AllSimdMaskTypes>, is_non_scalar>;
