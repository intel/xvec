//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <bit>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xvec/detail/compat.hpp>

namespace _XVEC_NAMESPACE::simd {

/// An alias for a suitable signed integer type to represent the size of a vec, index of an element, and so on.
using simd_size_type = int;

/// An alias for a compile time size constant.
template <simd_size_type _Np> using size_constant = std::integral_constant<simd_size_type, _Np>;

namespace detail
{

/// Compile time flag to indicate whether the LLVM compiler is being used.
#if defined(__llvm__)
inline constexpr bool isLlvm = true;
#else
inline constexpr bool isLlvm = false;
#endif

/// @brief Allow overloaded lambdas to be used for target-specific code.
/// Described at https://www.cppstories.com/2019/02/2lines3featuresoverload.html/
/// Working example at: https://godbolt.org/z/cY8nb3Gd6
// :TODO: Provide a default which asserts if no suitable case can be found.
template<class... Ts> struct target_overloads : Ts... { using Ts::operator()...; };
template<class... Ts> target_overloads(Ts...) -> target_overloads<Ts...>;

/// Detect whether a given type is std::complex
template <typename _Tp>
concept complex_number = std::is_same_v<std::remove_cvref_t<_Tp>, std::complex<typename _Tp::value_type>>;

/// Small helper: is T the same (after cvref-stripping) as any of Us...?
/// Used by mask_operator and other "T must be one of these" concepts.
template<class T, class... Us>
inline constexpr bool any_same_v = (std::same_as<std::remove_cvref_t<T>, Us> || ...);

/// Detect a fundamental unsigned integer type.
template<typename _Tp>
concept fundamental_unsigned_integer = any_same_v<_Tp,
                                                 unsigned char, unsigned short, unsigned int,
                                                 unsigned long, unsigned long long>;

/// Get the underlying compiler representation element type that will be used to
/// build the vec builtin type. Typically just the same type, but for complex
/// types it will be the type of the individual complex elements. For
/// user-defined types it will be the defined storage type.
///@{
template<typename _Tp> struct element_type { using value_type = _Tp; };
template<complex_number _Tp> struct element_type<_Tp> { using value_type = typename _Tp::value_type; };
template<typename _Tp> using element_type_t = typename element_type<_Tp>::value_type;
///@}

/// Get the value type of a complex type. If used for a non-complex type it will
/// return a dummy type.
///@{
template<typename _Tp> struct complex_element_type_query { struct value_type {}; }; // Dummy value type.
template<complex_number _Tp> struct complex_element_type_query<_Tp> { using value_type = typename _Tp::value_type; };
template<typename _Tp> using complex_element_type = typename complex_element_type_query<_Tp>::value_type;
///@}

/// Compile time query to determine whether the given type is a half-precision
/// float. Not every platform supports this so we can't use std::is_same
/// directly.
#if defined(__FLT16_MIN__)
template<typename _T> inline constexpr bool is_fp16_v = std::same_as<std::remove_cvref_t<_T>, _Float16>;
#else
template<typename _T> inline constexpr bool is_fp16_v = false;
#endif

template <typename _T> struct is_fp16 : std::bool_constant<is_fp16_v<_T>> {};

/// Define a utility to provoke a static_assert failure in certain unreachable constexpr statements. It can
/// be used like this:
///
/// if constexpr (someCompileTimeExpr) doSomething(); else vec_unreachable();
///
/// A plain static_assert wouldn't work there (prior to C++23) as it would always be evaluated.
/// @internal
template <typename>
inline constexpr bool dependent_false = false;

/// Utility to convert a concrete type into a dependent type. This defers type
/// checking from template definition time to instantiation time, allowing use
/// of forward-declared types in template functions. The type will be fully
/// defined by instantiation time. Usage:
///
///   template<typename T>
///   void func() {
///     dependent_type<T, ForwardDeclaredType<int>> obj;
///   }
///
/// @tparam Parent The template parameter to create dependency on
/// @tparam Type The concrete type to make dependent
template<typename _Parent, typename _Type>
using dependent_type = typename std::conditional<true, _Type, _Parent>::type;

/// Return an unsigned integer container which is the same size as the supplied type. The container
/// is a plain bucket-of-bits.
// :TODO: :COMPILER: The 128-bit type is required for std::complex<double>, but it isn't very efficient. Need to spend
// some time refining the code it generates to avoid the worst inefficiencies.
using containers_for_num_bytes = std::tuple<void, std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, __uint128_t>;

template<std::size_t _Bytes>
using container_for_num_bytes = typename std::tuple_element_t<std::bit_width(std::bit_ceil(_Bytes)), containers_for_num_bytes>;

template<typename _Tp> using container_for_type = container_for_num_bytes<sizeof(_Tp)>;

/// Temporary floating-point test. Used to handle half/_Float16 until the compiler catches up.
template<class _Tp>
struct is_floating_point
     : std::disjunction<std::is_floating_point<_Tp>, is_fp16<_Tp>> {};

// Convenience variable template + cvref stripping for our definition of floating point.
template<class T>
inline constexpr bool is_floating_point_v = is_floating_point<std::remove_cvref_t<T>>::value;

// Our own version of arithmetic to deal with FP16 properly.
template<class T>
inline constexpr bool is_arithmetic_like_v = std::is_integral_v<std::remove_cvref_t<T>> || is_floating_point_v<T>;

/// Return a signed version for the given type. Can't use std::make_unsigned_t directly
/// because it doesn't handle 128-bits properly.
template<typename _Tp> struct unsigned_to_signed { using value_type = std::make_signed_t<_Tp>; };
template<> struct unsigned_to_signed<unsigned __int128> { using value_type = signed __int128; };
template<typename _Tp> using make_signed_t = typename unsigned_to_signed<_Tp>::value_type;

/// The maximum native size of a register containing the given types.
template <typename _Tp> inline constexpr simd_size_type max_native_size = detail::maxBytesInVec / sizeof(_Tp);

/// The implementation-defined upper limit on size. Currently constrainted by
/// how big a compact mask can be (__int128_t).
template <typename _Tp> inline constexpr simd_size_type max_fixed_size = 128;

/// Get the type used to store the given element type. Typically just the same
/// type, but for complex types it will be the type of the individual complex
/// elements. Arithmetic types uses the obvious type, complex uses the
/// value_type of the complex, enums use their underlying type, and anything
/// else is mapped to a raw container.
///@{
template<typename _Tp> struct element_storage { using value_type = container_for_type<_Tp>; };
template<complex_number _Tp> struct element_storage<_Tp> { using value_type = typename _Tp::value_type; };
template<typename _Tp> requires (std::is_enum_v<_Tp>) struct element_storage<_Tp> { using value_type = std::underlying_type_t<_Tp>; };
template<typename _Tp> requires (std::is_arithmetic_v<_Tp> || is_floating_point_v<_Tp>) struct element_storage<_Tp> { using value_type = _Tp; };
template<typename _Tp> using element_storage_t = typename element_storage<_Tp>::value_type;
///@}

/// Query the extent of a given type.
template<typename _Tp>
constexpr auto get_span_static_extent(const _Tp& in) {
  if constexpr (requires { std::span{in}; })
    // Span is useful for extracting static extents from other things, like C-arrays or std::array.
    return std::integral_constant<std::size_t, decltype(std::span{in})::extent>{};
  else
    return std::integral_constant<std::size_t, std::dynamic_extent>{};
}

template<class _Span> inline constexpr std::size_t span_extent_v = decltype(get_span_static_extent(std::declval<_Span>()))::value;

/// Query the element type of a range or span-like type. Returns nullptr type if
/// the element type cannot be determined.
template<typename _Span>
constexpr auto get_span_element_type_helper(const _Span& in) {
  if constexpr (requires { std::span{in}; })
    // Span is useful for extracting static extents from other things, like C-arrays or std::array.
    return typename decltype(std::span{in})::value_type();
  else
    return nullptr;
}

template<class _Span> using span_element_type = decltype(get_span_element_type_helper(std::declval<_Span>()));

/// Enable or disable memory range checking for `unchecked' (not default init)
/// load/store operations. If enabled by the macro this checks that the memory
/// operation doesn't fall outside the valid range, throwing a range error if
/// such a bounds violation is detected.
template<typename _Vec, std::size_t _Extent, typename _Mask = bool>
constexpr void checkStaticMemoryBounds([[maybe_unused]] const std::string& op_name,
                                       [[maybe_unused]] std::span<typename _Vec::value_type, _Extent> mem_span,
                                       [[maybe_unused]] _Mask m = _Mask())
{
  constexpr bool opIsMasked = !std::same_as<_Mask, bool>;

  // There is a precondition that the span is big enough to do the memory
  // operation. However, if the span is a known size this can be explicitly
  // checked at compile time. Note that this check is only performed when no
  // mask is in use because the programmer may creating a mask to limit the
  // operation to valid memory. Without knowing what the mask is we can't tell
  // whether the precondition has been met or not. :TODO: If the mask is
  // constexpr, check that too.
  static_assert(opIsMasked || _Vec::size <= _Extent,
                "Memory span is not big enough for simd load/store operation");

#if defined(XVEC_ALWAYS_RANGE_CHECK_MEMORY)
  auto opSize = _Vec::size;

  if constexpr (opIsMasked) {
    // A mask has been provided. The caller may be using the mask to limit the
    // range. Check that no bits outside the valid range are set.
    auto outOfRangeBits = !_Mask::__mask_from_count(mem_span.size());
    if (any_of(outOfRangeBits & m))
    {
      std::string msg =
        "xvec masked range error for " + op_name + ". Range size:" + std::to_string(mem_span.size()) +
        " Mask:" + m.to_bitset().to_string();
      throw std::out_of_range(msg);
    }
  }
  else if (opSize > mem_span.size()) {
    std::string msg =
      "xvec range error for " + op_name + ". Range size:" + std::to_string(mem_span.size()) +
      " simd size:" + std::to_string(opSize);
    throw std::out_of_range(msg);
  }
#endif
}

/// Enable or disable memory range checking for `unchecked' indexed operations
/// (gather, scatter). If enabled by the macro this checks that the memory
/// operation doesn't fall outside the valid range, throwing a range error if
/// such a bounds violation is detected.
template<typename _Vec>
constexpr void checkStaticMemoryBounds([[maybe_unused]] const std::string& op_name,
                                       [[maybe_unused]] const _Vec& indexes,
                                       [[maybe_unused]] std::size_t range_size,
                                       [[maybe_unused]] typename _Vec::mask_type m = typename _Vec::mask_type(true))
{
#if defined(XVEC_ALWAYS_RANGE_CHECK_MEMORY)
  auto biggestIndex = reduce_max(indexes, m);
  if (biggestIndex >= range_size)
  {
    std::string msg =
      "xvec range error for " + op_name + ". Range size:" + std::to_string(range_size) +
      " maxIndex:" + std::to_string(biggestIndex);
    throw std::out_of_range(msg);
  }
#endif
}

// Create a constant_wrapper-like variable which can be used to create
// compile-time constants. Eventually this will be superseded by C++26's own cw handling.
template <int X> inline constexpr std::integral_constant<int, X> cw{};

} // End of namespace detail

} // namespace _XVEC_NAMESPACE::simd

/* Figure out whether to define the output operators */
#if defined(_IOSTREAM_) || defined(_CPP_IOSTREAM) ||                           \
    defined(_GLIBCXX_IOSTREAM) || defined(_LIBCPP_IOSTREAM)
#define _XVEC_DEFINE_OUTPUT_OPERATORS
#endif
