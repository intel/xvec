//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <bitset>
#include <concepts>     // std::same_as, std::convertible_to, std::integral, ...
#include <cstddef>      // std::size_t
#include <cstring>      // std::memcpy
#include <algorithm>    // std::min, std::max
#include <bit>          // std::has_single_bit, std::bit_ceil
#include <functional>   // std::less, std::greater, ... (used by mask_operator)
#include <limits>       // std::numeric_limits
#include <ranges>       // std::ranges::contiguous_range, sized_range
#include <type_traits>
#include <utility>      // std::index_sequence, std::in_range

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>

namespace _XVEC_NAMESPACE::simd {

/// @brief An implementation ABI type which is used to specify a simd::vec with a fixed number of elements
/// @tparam _Np The compile-time number of elements to use in a simd::vec type.
/// @tparam _TARGET_TAG A defaulted target tag which is used to ensure that
/// inline or constexpr functions compiled for different target tags will
/// resolve to different mangled function names.
template<simd_size_type _Np, typename _TARGET_TAG = target_tag> struct simd_fixed_size_abi {
  static constexpr simd_size_type num_elements = _Np; ///< Requested number of elements
};

template<typename _Tp,       typename _Abi = simd_fixed_size_abi<detail::maxBytesInVec / sizeof(_Tp)>> class basic_vec;
template<std::size_t _Bytes, typename _Abi = simd_fixed_size_abi<detail::maxBytesInVec / _Bytes>>      class basic_mask;

/// Special ABI type used to indicate that the native register size should be used.
template<typename _Tp> using simd_native_abi = simd_fixed_size_abi<detail::max_native_size<_Tp>>;

/// Query how many elements are contained in a basic_vec<_Tp, _Abi>.
// :TODO: Should be zero on unsupported types.
template<class _Tp, class _Abi = simd_native_abi<_Tp>> inline constexpr simd_size_type simd_size_v = _Abi::num_elements;

/// Query the number of bytes in an element of a simd::mask.
template<class _Tp> struct simd_mask_element_size : std::integral_constant<std::size_t, 0> {};
template<std::size_t _Bytes, typename _Abi> struct simd_mask_element_size<basic_mask<_Bytes, _Abi>> : std::integral_constant<std::size_t, _Bytes> {};
template <typename _Tp> inline constexpr std::size_t simd_mask_element_size_v = simd_mask_element_size<_Tp>::value;

namespace detail {
  template <typename> inline constexpr bool is_vec_v = false;
  template <typename _Tp, typename _Abi> inline constexpr bool is_vec_v<basic_vec<_Tp, _Abi>> = true;

  template <typename> inline constexpr bool is_mask_v = false;
  template <std::size_t _Bytes, typename _Abi> inline constexpr bool is_mask_v<basic_mask<_Bytes, _Abi>> = true;
}

template<typename _Vp> concept vec_type = detail::is_vec_v<std::remove_cvref_t<_Vp>>;
template<typename _Mp> concept mask_type = detail::is_mask_v<std::remove_cvref_t<_Mp>>;
template<typename _Vp> concept vec_or_mask_type = vec_type<_Vp> || mask_type<_Vp>;

template<typename _V> concept vec_integral =          vec_type<_V> && std::integral<typename _V::value_type>;
template<typename _V> concept vec_signed_integral =   vec_type<_V> && std::signed_integral<typename _V::value_type>;
template<typename _V> concept vec_unsigned_integral = vec_type<_V> && std::unsigned_integral<typename _V::value_type>;
template<typename _V> concept vec_unsigned_integer =  vec_type<_V> && detail::fundamental_unsigned_integer<typename _V::value_type>;
template<typename _V> concept vec_floating_point =    vec_type<_V> && detail::is_floating_point_v<typename _V::value_type>;
template<typename _V> concept vec_complex =           vec_type<_V> && detail::complex_number<typename _V::value_type>;
template<typename _V> concept vec_arithmetic =        vec_integral<_V> || vec_floating_point<_V>;
template<typename _V> concept vec_totally_ordered = vec_type<_V> && std::totally_ordered<typename _V::value_type>;

/// Detect a vec_type with arbitrary length whose elements are of a given type.
template<typename _Vp, typename _Tp> concept vec_of = vec_type<_Vp> && std::same_as<typename _Vp::value_type, _Tp>;

/// Check that two simdable values (mask, simds, or a mixture) have the same number of elements.
template<typename _V0, typename _V1> concept simd_same_size = vec_or_mask_type<_V0> && vec_or_mask_type<_V1> && (_V0::size == _V1::size);

/// Check that the two simdable values have the same type/size of elements.
template<typename _V0, typename _V1> concept vec_or_mask_same_element_type =
    (vec_type<_V0> && vec_type<_V1> && std::same_as<typename _V0::value_type, typename _V1::value_type>) ||
    (mask_type<_V0> && mask_type<_V1> && simd_mask_element_size_v<_V0> == simd_mask_element_size_v<_V1>);

namespace detail
{

// Concepts to check whether a type is a promotable (i.e., where T op T gives something bigger).
template<class _Tp>
concept promotable_type =
  std::is_arithmetic_v<_Tp> ||
  (std::is_enum_v<_Tp> && !detail::is_scoped_enum<_Tp>);

// The op must exist; and if _Tp is not promotable then the result type must be
// exactly _Tp (no implicit promotions allowed).
template<class _Tp, class _Op>
concept supported_unary_op =
  requires(_Tp a) { _Op{}(a); } &&
  (promotable_type<_Tp> || requires(_Tp a) { { _Op{}(a) } -> std::same_as<_Tp>; });

template<class _Tp, class _Op>
concept supported_binary_op =
  requires(_Tp a, _Tp b) { _Op{}(a, b); } &&
  (promotable_type<_Tp> || requires(_Tp a, _Tp b) { { _Op{}(a, b) } -> std::same_as<_Tp>; });

template<class _Tp, class _Op>
concept supported_relational_op = (requires(_Tp a, _Tp b) { {_Op{}(a, b)} -> std::same_as<bool>; });


/// Concept to determine whether a value of one scalar type can be converted
/// to another in a way which preserves every possible value.
template<typename _From, typename _To>
inline constexpr bool value_preserving_convertible_to_v = []{
  using BareFrom = std::remove_cvref_t<_From>;
  using BareTo   = std::remove_cvref_t<_To>;

  if constexpr (std::same_as<BareFrom, BareTo>)
    return true;
  else if constexpr (complex_number<BareTo>)
  {
    using ToComponent = typename BareTo::value_type;
    if constexpr (complex_number<BareFrom>)
      return value_preserving_convertible_to_v<typename BareFrom::value_type, ToComponent>;
    else
      return value_preserving_convertible_to_v<BareFrom, ToComponent>;
  }
  else if constexpr (std::integral<BareFrom> && std::integral<BareTo>)
  {
    if constexpr (std::is_signed_v<BareFrom> == std::is_signed_v<BareTo>)
      return (sizeof(BareFrom) <= sizeof(BareTo));
    else if constexpr (std::is_unsigned_v<BareFrom> && std::is_signed_v<BareTo>)
      return (sizeof(BareFrom) < sizeof(BareTo));
    else
      return false;
  }
  else if constexpr (std::integral<BareFrom> && is_floating_point_v<BareTo>)
    return std::numeric_limits<BareTo>::digits >= std::numeric_limits<BareFrom>::digits;
  else if constexpr (is_floating_point_v<BareFrom> && is_floating_point_v<BareTo>)
    return (std::numeric_limits<BareTo>::digits       >= std::numeric_limits<BareFrom>::digits &&
            std::numeric_limits<BareTo>::max_exponent >= std::numeric_limits<BareFrom>::max_exponent &&
            std::numeric_limits<BareTo>::lowest()     <= std::numeric_limits<BareFrom>::lowest() &&
            std::numeric_limits<BareTo>::max()        >= std::numeric_limits<BareFrom>::max());
  else
    return false;
}();

template<typename _From, typename _To>
concept value_preserving_convertible_to = value_preserving_convertible_to_v<_From, _To>;

// std::convertible_to only checks implicit conversions. This concept also
// allows explicit conversions (e.g., static_cast), which is needed for types
// that have explicit constructors or explicit conversion operators.
template<class _From, class _To> concept explicitly_convertible_to =
    requires {
      static_cast<_To>(std::declval<_From>());
    };

template <typename _From, typename _To>
concept higher_floating_point_rank = is_floating_point_v<_From> && is_floating_point_v<_To>
                                     && std::same_as<std::common_type_t<_From, _To>, _From>
                                     && !std::same_as<_From, _To>;

/// This implementation cheats by assuming the size indicates the rank. For
/// practical purposes this is generally sufficient, but it will fall short of
/// the complicated rules for rank conversion in the standard.
template <typename _From, typename _To>
concept higher_integer_rank = std::integral<_From> && std::integral<_To> && (sizeof(_From) > sizeof(_To));

/// Check for contiguous range with a finite range.
template<typename _Range> concept contiguous_sized_range = std::ranges::contiguous_range<_Range> && std::ranges::sized_range<_Range>;

/// Detect a simple vectorizable type.
template<typename _Tp>
concept simple_vectorizable =
  sizeof(_Tp) <= 8 &&
  ((std::integral<_Tp> && !std::same_as<bool, _Tp>) || 
    is_floating_point_v<_Tp> ||
    std::same_as<std::byte, _Tp>);

/// Detect an extended vectorizable type. These are arithmetic types which aren't defined
/// in the standard, but which work for this implementation.
template<typename _Tp>
concept extended_vectorizable = any_same_v<_Tp, signed __int128, unsigned __int128>;

/// Detect a type which has builtin support (i.e., the library explicitly
/// handles operations on it). A pre-defined list of types whose behaviour is
/// fixed.
template<typename _Tp>
concept builtin_vectorizable =
  simple_vectorizable<_Tp>
  || extended_vectorizable<_Tp>
  || (complex_number<_Tp> && simple_vectorizable<typename _Tp::value_type>);

/// Trait-based form of vectorization concept.
template<typename _Tp>
inline constexpr bool disable_vectorization =
    std::is_pointer_v<_Tp> ||
    std::is_member_pointer_v<_Tp> ||
    std::is_union_v<_Tp> ||
    std::is_const_v<_Tp> ||
    std::is_volatile_v<_Tp> ||
    std::is_empty_v<_Tp> ||
    std::same_as<_Tp, long double> ||
    vec_type<_Tp>;
    // Users can specialize this for their own types
    // Implementations add specializations for specific std types

template<typename _Tp>
concept vectorizable =
    std::is_trivially_copyable_v<_Tp> &&
    (sizeof(_Tp) == 1 || sizeof(_Tp) == 2 || sizeof(_Tp) == 4 ||
     sizeof(_Tp) == 8 || sizeof(_Tp) == 16) &&
    !disable_vectorization<_Tp>;

template<typename _Tp>
concept udt_vectorizable = vectorizable<_Tp> && !builtin_vectorizable<_Tp>;

/// Determine whether a simd element conversion from _From to _To must be
/// explicit. For builtin vectorizable types (arithmetic, __int128, complex),
/// this uses the traditional value-preserving and rank-based rules from
/// [simd.ctor]. For all other types (enums, UDTs, or mixed builtin/UDT
/// conversions), P2964R2 specifies that simd defers to the type author's
/// judgment: if the scalar conversion is implicit (is_convertible_v is true),
/// the simd conversion is implicit; if the scalar conversion requires explicit
/// construction, so does the simd conversion.
template<typename _From, typename _To>
inline constexpr bool needs_explicit_conversion_v = [] {
  if constexpr (builtin_vectorizable<_From> && builtin_vectorizable<_To>)
    return !value_preserving_convertible_to<_From, _To> ||
           higher_floating_point_rank<_From, _To> ||
           higher_integer_rank<_From, _To>;
  else
    return !std::is_convertible_v<_From, _To>;
}();

template<typename _From, typename _To>
concept needs_explicit_conversion = needs_explicit_conversion_v<_From, _To>;

/// Concept to detect a type which is like a constexpr wrapper (i.e., it has a
/// static member named `value` which is the same as the default value of the
/// type, and which can be converted to the underlying type).
template<class T>
concept constexpr_wrapper_like =
  requires { T::value; } &&
  requires {
    requires std::convertible_to<T, std::remove_cvref_t<decltype(T::value)>>;
    requires std::equality_comparable_with<T, std::remove_cvref_t<decltype(T::value)>>;
    requires std::bool_constant<(T{} == T::value)>::value;
    requires std::bool_constant<
      (static_cast<std::remove_cvref_t<decltype(T::value)>>(T{}) == T::value)
    >::value;
  };

template<class From, class To>
concept roundtrippable =
  std::is_convertible_v<From, To> &&
  std::is_convertible_v<To, From> &&
  requires(From x) {
    { static_cast<From>(static_cast<To>(x)) == x } -> std::convertible_to<bool>;
  };

template<class To, class From>
requires roundtrippable<From, To>
constexpr bool roundtrip_preserves_value(From x) { return static_cast<From>(static_cast<To>(x)) == x; }

template<class Wrapper, class To>
inline constexpr bool wrapped_value_is_representable_by_v = []{
  using From = std::remove_cvref_t<decltype(Wrapper::value)>;

  if constexpr (std::is_integral_v<From> && std::is_integral_v<To>) {
    return std::in_range<To>(Wrapper::value);
  } else if constexpr (roundtrippable<From, To>) {
    return roundtrip_preserves_value<To>(Wrapper::value);
  } else {
    return false;
  }
}();

template<class Wrapper, class To>
concept wrapped_value_is_representable_by = wrapped_value_is_representable_by_v<Wrapper, To>;

template<class From, class To>
concept broadcast_constructible =
  std::convertible_to<From, To> &&
  (
    // 1) From is not arithmetic-like and not a constexpr-wrapper-like
    (!is_arithmetic_like_v<From> && !constexpr_wrapper_like<From>)

    // 2) From is arithmetic-like and conversion From -> value_type is value-preserving
    || (is_arithmetic_like_v<From> && value_preserving_convertible_to<From, To>)

    // 3) From is constexpr-wrapper-like, its ::value type is arithmetic-like,
    //    and that value is representable by value_type
    || (constexpr_wrapper_like<From> &&
          is_arithmetic_like_v<std::remove_cvref_t<decltype(From::value)>> &&
          wrapped_value_is_representable_by<From, To>)
  );

template<typename _Gen> concept index_generator_function_with_size = requires { { std::declval<_Gen>()(0, 0) } -> std::integral; };
template<typename _Gen> concept index_generator_function_without_size = requires { { std::declval<_Gen>()(0) } -> std::integral; };
template<typename _Gen> concept index_generator_function = index_generator_function_with_size<_Gen> || index_generator_function_without_size<_Gen>;

/// Concepts used to simplify some function signature requirements.
template<typename _Tp> concept vec_container_element = std::unsigned_integral<_Tp> || std::same_as<unsigned __int128, _Tp>;

/// Concepts to check that generators are correctly invocable. The value returned for every possible
/// call to the generator in the range [0..size) must produce a value which can be stored in a vec.
///@{
template<typename _From, typename _To>
concept generated_value_convertible_to =
  std::convertible_to<_From, _To> &&
  (!is_arithmetic_like_v<_From> || // !arithmetic
    detail::value_preserving_convertible_to<_From, _To>);

template<typename _Gen, typename _Tp, simd_size_type _Idx>
concept generator_invocable_at =
  requires(_Gen __gen) {
    { __gen(_Idx) } -> generated_value_convertible_to<_Tp>;
  };

template<typename _Gen, typename _Tp, int _Size>
concept generator_invocable =
  !std::constructible_from<_Tp, _Gen> && // if the generator can be used as a constructor argument, it shouldn't be used as a generator
  []<simd_size_type... _Idx>(std::integer_sequence<simd_size_type, _Idx... >) {
    // Note - avoiding folds because older compilers have size limitations with them.
    bool checks[] = { generator_invocable_at<_Gen, _Tp, _Idx>... };
    for (bool check : checks) if (!check) return false;
    return true;
  }(std::make_integer_sequence<simd_size_type, _Size>{});
///@}

template<typename Op>
concept mask_operator = any_same_v<Op,
  std::less<>, std::greater<>, std::less_equal<>, std::greater_equal<>,
  std::equal_to<>, std::not_equal_to<>,
  std::logical_and<>, std::logical_or<>, std::logical_not<>>;

// [simd.expos.defn]
template <class _Tp> using deduced_vec_t = decltype(std::declval<const _Tp&>() + std::declval<const _Tp&>());
template <class _Tp> using deduced_mask_t = typename deduced_vec_t<_Tp>::mask_type;
template<class T> concept math_floating_point = vec_floating_point<deduced_vec_t<T>>;

} // namespace detail

/// Given a simd type (vec or mask), create a new type which has the same size but a different element.
template<class _Tp, class _Vp> struct rebind {};
template<class _Tp, class _Sp, class _Abi> struct rebind<_Tp, basic_vec<_Sp, _Abi>> {
  using type = basic_vec<_Tp, simd_fixed_size_abi<_Abi::num_elements>>;
};
template<class _Tp, std::size_t _Bytes, class _Abi> struct rebind<_Tp, basic_mask<_Bytes, _Abi>> {
  using type = basic_mask<sizeof(_Tp), simd_fixed_size_abi<_Abi::num_elements>>;
};
template<class _Tp, class _Vp> using rebind_t = typename rebind<_Tp, _Vp>::type;

/// Given a vec or mask type, create a new type which has the same element type but a different size.
template<simd_size_type _Np, class _Vp> struct resize { };
template<simd_size_type _Np, class _Tp, class _Abi> struct resize<_Np, basic_vec<_Tp, _Abi>> {
  using type = basic_vec<_Tp, simd_fixed_size_abi<_Np>>;
};
template<simd_size_type _Np, std::size_t _Bytes, class _Abi> struct resize<_Np, basic_mask<_Bytes, _Abi>> {
  using type = basic_mask<_Bytes, simd_fixed_size_abi<_Np>>;
};
template<simd_size_type _Np, class _Vp> using resize_t = typename resize<_Np, _Vp>::type;

/// Shortened aliases for common types.
template<class _Tp, simd_size_type _Np = basic_vec<_Tp>::size()> using vec = basic_vec<_Tp, simd_fixed_size_abi<_Np>>;
template<class _Tp, simd_size_type _Np = basic_mask<sizeof(_Tp)>::size()> using mask = basic_mask<sizeof(_Tp), simd_fixed_size_abi<_Np>>;

///@{
/// flags for controlling the behaviour of simd operations like conversions and memory accesses.

template <class... _Flags> struct flags {
  template <class... _Others>
  friend consteval auto operator|(flags, flags<_Others...>) {
    // :TODO: Remove duplicates? What happens if overaligned is specified multiple times?
    return flags<_Flags..., _Others...>();
  }
};

namespace detail {
  class flag_convert {};
  class flag_aligned {};
  template<size_t _N> class flag_overaligned {};
  class flag_unchecked {};

  // Add extra flags to the existing flags.
  ///@{
  constexpr auto add_unchecked_flag(auto f) { return f | flags<flag_unchecked>{}; };
  constexpr auto add_convert_flag(auto f) { return f | flags<flag_convert>{}; };
  ///@}

  /// @brief Query whether the given flags contains the specified type.
  template<typename _F, typename... _Flags>
  constexpr bool contains_flag(flags<_Flags...>) { return (std::is_same_v<_F, _Flags> || ...); }
}

/// Standard flags.
///@{
inline constexpr flags<> flag_default{};
inline constexpr flags<detail::flag_convert> flag_convert{};
inline constexpr flags<detail::flag_aligned> flag_aligned{};

template<size_t _N> requires (std::has_single_bit(_N)) inline constexpr flags<detail::flag_overaligned<_N>> flag_overaligned{};
///@}

/// Conversion flag, used to pass destination type to a simd_convert customisation point.
template<typename _Tp>
struct convert_to_t {
  using type = _Tp;
  constexpr explicit convert_to_t() noexcept = default;
};

template<class _Tp> inline constexpr convert_to_t<_Tp> convert_to{};

namespace detail
{

/// Traits for standard builtin vector types.
template<typename _Vendor, typename _Tp, simd_size_type _Np>
struct vec_traits
{
  static_assert(isLlvm || _Np <= 256, "gcc has limits on size");

  /// The number of underlying builtin data elements.
  static constexpr simd_size_type _numDataElements = complex_number<_Tp> ? 2 * _Np : _Np;

  /// The number of storage elements. This could be bigger than the number of data elements.
  static constexpr simd_size_type _numStorageElements = isLlvm ? _numDataElements : std::bit_ceil((uint64_t)_numDataElements);

  /// The bare storage type used by the compiler for the elements. For
  /// arithmetic types this will typically be the same as the type itself, but
  /// for user-defined types or complex types it might be a container of raw
  /// bits instead.
  using element_type = element_storage_t<_Tp>;
  using element_container_type = container_for_type<_Tp>;

  /// The compiler's representation of the simd::vec type.
  using builtin_type __attribute__((__vector_size__(sizeof(element_type) * _numStorageElements))) = element_type;

  /// @brief The best register size to use for operations on this builtin type.
  using register_type __attribute__((__vector_size__(sizeof(element_type) * register_size<element_type, _numDataElements>))) = element_type;

  /// @brief Allow the builtin data to be turned into the closest matching
  /// target register type. This will return the smallest target register that
  /// can contain all the elements of the builtin type. It is an error to try to
  /// store the data in a register which is too small. In that case the user
  /// must break the builtin type into pieces which do fit in a register (e.g.,
  /// using fit_to_size, or extract). If the builtin type contains less data
  /// than the register then the unused register elements will have undefined
  /// values.
  static constexpr register_type to_register(builtin_type v) {
    // Use a static assert rather than a requires to allow this error to be
    // reported instead of rather than resulting in a confusing lookup failure.
    constexpr auto _regSize =  register_size<_Tp, _Np>;
    static_assert(_Np <= _regSize, "Converting to a smaller register loses data");
    if constexpr (_Np == _regSize)
      return v;
    else
    {
      return [=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
        return __builtin_shufflevector(v, v, (_Idx < _Np ? int(_Idx) : -1)...);
      }(std::make_index_sequence<_regSize>());
    }
  }

  /// @brief A named constructor for creating a builtin type from a register value.
  /// @param r The register value from which to create the builtin value.
  /// @return A builtin type containing the values from the register.
  static constexpr builtin_type from_register(register_type r) {
    constexpr auto _regSize = register_size<_Tp, _Np>;
    static_assert(_Np <= _regSize, "Converting from a smaller register creates undefined values");
    if constexpr (_Np == _regSize)
      return r;
    else
    {
      return [=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
        return __builtin_shufflevector(r, r, ((_Idx < _numDataElements) ? int(_Idx) : -1)...);
      }(std::make_index_sequence<_numStorageElements>());
    }
  }

};

/// Traits for generic mask types which use a standard vector register to represent the mask.
template<typename _Vendor, std::size_t _Bytes, simd_size_type _Np>
struct mask_traits
{
  using element_container_type = detail::container_for_num_bytes<_Bytes>;
  using element_signed_container_type = detail::make_signed_t<element_container_type>;

  using signed_vec_for_mask = vec<element_signed_container_type, _Np>;
  using unsigned_vec_for_mask = vec<element_container_type, _Np>;

  // Round up to the next power of 2 to ensure efficient storage. Although
  // clang/icx handle arbitrary sizes, they don't handle non-byte sizes very
  // well, and introduce lots of unnecessary code.
  static constexpr int internalNp = std::bit_ceil(std::max<unsigned>(8, _Np));

#if defined (__llvm__)
  #if (__INTEL_LLVM_COMPILER > 20220000)
    using compact_builtin_type = unsigned _BitInt(internalNp);
  #else
    using compact_builtin_type = unsigned _ExtInt(internalNp);
  #endif

  static constexpr auto all_bits_set = (_Np == internalNp) ? ~compact_builtin_type() : ((compact_builtin_type(1) << _Np) - 1);

#else
  using compact_builtin_type = __uint128_t;
  static constexpr auto all_bits_set = (_Np == 128) ? ~compact_builtin_type() : ((compact_builtin_type(1) << _Np) - 1);
#endif

  using builtin_type =
    std::conditional_t<has_compact_mask<_Vendor>, compact_builtin_type, typename vec_traits<_Vendor, element_container_type, _Np>::builtin_type>;

  static constexpr builtin_type truncate_to_size(builtin_type m) {
    if constexpr (has_compact_mask<_Vendor>) return m & all_bits_set;
    else return m;
  }

  static constexpr auto to_register(const builtin_type& m) noexcept {
    if constexpr (has_compact_mask<_Vendor>) return m;
    else return vec_traits<_Vendor, element_container_type, _Np>::to_register(m);
  }
};

} // namespace detail

} // namespace _XVEC_NAMESPACE::simd
