//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <bitset>
#include <type_traits>
#include <cstddef>
#include <concepts>
#include <functional>

#include <iostream>

#include <xvec/detail/utilities.hpp>

namespace _XVEC_NAMESPACE::simd
{

/**
 * @defgroup cpp26_mask_overview C++26: Mask overview
 * @brief Mask types, element counts, special members, and range access.
 *
 * @see https://eel.is/c++draft/simd.mask.overview
 */

/**
 * @defgroup cpp26_mask_ctor C++26: Mask constructors
 * @brief Construct masks from boolean values, generators, bitsets, integers, and other masks.
 *
 * @see https://eel.is/c++draft/simd.mask.ctor
 */

/**
 * @defgroup cpp26_mask_conv C++26: Mask conversions
 * @brief Convert masks to vectors and packed-bit representations.
 *
 * @see https://eel.is/c++draft/simd.mask.conv
 */

/**
 * @defgroup cpp26_mask_subscr C++26: Mask subscripting
 * @brief Access individual mask elements or select them using an index vector.
 *
 * @see https://eel.is/c++draft/simd.mask.subscr
 */

/**
 * @defgroup cpp26_mask_unary C++26: Unary mask operators
 * @brief Apply unary operations to mask elements.
 *
 * @see https://eel.is/c++draft/simd.mask.unary
 */

/**
 * @defgroup cpp26_mask_binary C++26: Binary mask operators
 * @brief Apply logical and bitwise operations to corresponding mask elements.
 *
 * @see https://eel.is/c++draft/simd.mask.binary
 */

/**
 * @defgroup cpp26_mask_cassign C++26: Mask compound assignment
 * @brief Apply bitwise operations and assign their results.
 *
 * @see https://eel.is/c++draft/simd.mask.cassign
 */

/**
 * @defgroup cpp26_mask_comparison C++26: Mask comparisons
 * @brief Compare corresponding mask elements.
 *
 * @see https://eel.is/c++draft/simd.mask.comparison
 */

/**
 * @defgroup cpp26_mask_cond C++26: Exposition-only mask selection
 * @brief Implementation helpers for element-wise conditional selection.
 *
 * @see https://eel.is/c++draft/simd.mask.cond
 */

/**
 * @defgroup cpp26_mask_reductions C++26: Mask reductions
 * @brief Query mask elements, count selected elements, and find their indexes.
 *
 * @see https://eel.is/c++draft/simd.mask.reductions
 */

/**
 * @defgroup cpp29_p3440 Create mask from count
 * @brief Create a mask with a specified number of leading true elements.
 *
 * Proposed in P3440 for uses such as handling loop remainders.
 *
 * @see https://wg21.link/P3440
 */

/// @brief A mask containing boolean elements.
/// @ingroup cpp26_mask_overview
/// @tparam _Bytes The associated vector element size in bytes.
/// @tparam _Abi The ABI determining the mask representation and element count.
/// @details Provides a uniform interface over target-specific mask representations.
///
/// Wide masks, such as those used with Intel SSE and AVX, represent each boolean
/// element using a vector element whose bits are all zero or all one.
///
/// Compact masks, such as those used with Intel AVX-512, represent each boolean
/// element using a single bit.
///
/// The class hides these representation differences for logical operations,
/// comparisons, selection, and conversion to packed-bit representations.
template<std::size_t _Bytes, typename _Abi>
class basic_mask
{
public:

  /// Implementation traits for this mask type.
  /// @internal
  using traits = detail::mask_traits<target_tag, _Bytes, _Abi::num_elements>;

  /// Compiler-specific storage representation of the mask.
  /// @ingroup xvec_interop
  using builtin_type = typename traits::builtin_type;

  /// @addtogroup cpp26_mask_overview
  /// @{

  /// The ABI type of this mask.
  using abi_type = _Abi;

  /// The boolean type of each mask element.
  using value_type = bool;

  /// Iterator over mask elements.
  using iterator = simd_iterator<basic_mask>;

  /// Iterator providing read-only access to mask elements.
  using const_iterator = simd_iterator<const basic_mask>;

  /// Return the number of elements contained in the mask.
  static constexpr auto size = size_constant<_Abi::num_elements>{};

  /// @} cpp26_mask_overview

private:

  /// The builtin mask value.
  /// @internal
  builtin_type mask;

public:

  /// @addtogroup cpp26_mask_overview
  /// @{

  /// Default-construct a mask.
  constexpr basic_mask() = default;

  /// Copy-construct a mask.
  constexpr basic_mask(const basic_mask&) = default;

  /// Move-construct a mask.
  constexpr basic_mask(basic_mask&&) noexcept = default;

  /// Copy-assign a mask.
  constexpr basic_mask& operator=(const basic_mask&) = default;

  /// Move-assign a mask.
  constexpr basic_mask& operator=(basic_mask&&) noexcept = default;

  /// @} cpp26_mask_overview

  /// Allow a mask to be constructed from the compiler's own representation. The
  /// compiler may use multiple or partial target registers in its
  /// representation. This constructor allows a mask to be built from the output
  /// of an overloaded operator, a builtin, and so on.
  /// @ingroup xvec_interop
  constexpr basic_mask(const builtin_type& m) noexcept : mask(m) {}

  /// @addtogroup cpp26_mask_ctor
  /// @{

  /// @brief Build a mask from a boolean value.
  /// @param b The boolean value to broadcast to all mask elements.
  constexpr explicit basic_mask(std::same_as<value_type> auto b) noexcept :
    mask([b] { if (b) return !basic_mask(); else return basic_mask(); }()) {}

  /// @brief Build a mask from the bits of an unsigned integral value.
  /// Each mask element is initialized from the corresponding input bit.
  /// Elements beyond the width of the input value are initialized to false.
  /// @param m The unsigned integral value supplying the bits.
  template <std::unsigned_integral T> requires (!std::same_as<T, value_type>)
  constexpr explicit basic_mask(T m) noexcept
    : basic_mask([m]() {
        if (std::is_constant_evaluated())
          return basic_mask([m](auto idx) { return (idx < std::numeric_limits<T>::digits) ? ((m >> idx) & 1) == 1 : 0; });
        else
          return detail::generate_mask_from_unsigned<basic_mask>(target, m);
    }()) {}

  /// @brief Convert a mask with the same number of elements.
  /// Each output element retains the corresponding input boolean value.
  /// @tparam _UBytes The source mask's associated vector element size in bytes.
  /// @tparam _UAbi The source mask ABI.
  /// @param m The mask to convert.
  template<std::size_t _UBytes, typename _UAbi>
  requires (basic_mask<_UBytes, _UAbi>::size() == size())
  explicit constexpr basic_mask(const basic_mask<_UBytes, _UAbi>& m) : mask(detail::convert_mask<basic_mask>(target, m)) {}

  /// @brief Generate each mask element from its index.
  /// @param fn The generator returning a boolean value for each element index.
  constexpr basic_mask(std::invocable<std::integral_constant<simd_size_type, 0>> auto fn) : basic_mask(detail::generate_mask<basic_mask>(target, fn)) {}

  /// @} cpp26_mask_ctor

  /// Generate a mask with the first _n elements set to true.
  /// @internal
  static constexpr basic_mask __mask_from_count(simd_size_type _n)
  {
    if (std::is_constant_evaluated())
      // get_n_bit_mask may have intrinsic (non-constexpr) calls so use a generator for the constexpr case.
      return basic_mask([&](simd_size_type i) { return i < _n; });
    else
      // x86 has special instructions which can accelerate mask generation.
      return detail::get_n_bit_mask(target, basic_mask(), _n);
  }

  /// @brief Construct a mask from the corresponding bitset.
  /// @ingroup cpp26_mask_ctor
  /// @param b The bitset supplying the value of each mask element.
  // :TODO: Not efficient if the bitset isn't constexpr. Need a better way to deal with that eventually.
  template<std::same_as<std::bitset<size()>> T>
  constexpr basic_mask(const T& b) noexcept : basic_mask([&](simd_size_type idx) { return b[idx]; }) {}

  /// @addtogroup cpp26_mask_overview
  /// @{

  /// Return an iterator to the first element.
  constexpr iterator begin() noexcept { return {*this, 0}; }

  /// Return a const iterator to the first element.
  constexpr const_iterator begin() const noexcept { return {*this, 0}; }

  /// Return a const iterator to the first element.
  constexpr const_iterator cbegin() const noexcept { return {*this, 0}; }

  /// Return the end sentinel.
  constexpr std::default_sentinel_t end() const noexcept { return {}; }

  /// Return the end sentinel.
  constexpr std::default_sentinel_t cend() const noexcept { return {}; }

  /// @} cpp26_mask_overview

  /// Return the underlying mask representation explicitly.
  /// @ingroup xvec_interop
  // :TODO: Maybe to_builtin is sufficient?
  constexpr explicit operator builtin_type() { return mask; }

  /// @brief Convert the mask to a signed integral vector.
  /// Each output element is 1 if the corresponding mask element is true,
  /// or 0 otherwise.
  /// @ingroup cpp26_mask_conv
  constexpr operator typename traits::signed_vec_for_mask() const { return detail::mask_to_vec<1, typename traits::element_signed_container_type>(target, *this); }

  /// @addtogroup xvec_interop
  /// @{

  /// Return the builtin mask value directly.
  constexpr auto to_builtin() const { return mask; }

  /// Construct a mask directly from its builtin representation.
  /// This named operation distinguishes representation conversion from
  /// construction using a public value type such as an unsigned integer.
  /// @param m The builtin mask representation.
  static constexpr basic_mask from_builtin(const builtin_type& m) { basic_mask bm; bm.mask = traits::truncate_to_size(m); return bm; }

  /// Convert the mask into a register which can be passed to an intrinsic.
  constexpr auto to_register() const noexcept { return traits::to_register(mask); }

  /// @} xvec_interop

  /// @addtogroup cpp26_mask_conv
  /// @{

  /// Return a bitset containing the value of each mask element.
  constexpr std::bitset<size()> to_bitset() const noexcept { return detail::mask_to_bitset(target, *this); }

  /// Return an unsigned integral representation of the mask elements.
  /// Elements beyond the width of unsigned long long must be false.
  constexpr unsigned long long to_ullong() const {
    if (std::is_constant_evaluated())
    {
      constexpr auto numOutputBits = std::numeric_limits<unsigned long long>::digits;

      if (any_of(*this) && reduce_max_index(*this) >= numOutputBits)
        throw "Mask has set bits which won't be copied into the output ullong";

      unsigned long long result = 0;
      for (simd_size_type i = 0; i < std::min(size(), numOutputBits); ++i)
        result |= static_cast<unsigned long long>((*this)[i]) << i;
      return result;
    }
    else
      return detail::mask_to_ullong(target, *this);
  }

  /// @} cpp26_mask_conv

  /// @addtogroup cpp26_mask_unary
  /// @{

  /// Negate each boolean mask element.
  constexpr basic_mask operator!() const noexcept { return detail::builtin_operator(target, *this, std::bit_not{}); }

  /// Return a signed integral vector containing 1 for true elements and 0 otherwise.
  constexpr typename traits::signed_vec_for_mask operator+() const noexcept { return detail::mask_to_vec<1, typename traits::element_signed_container_type>(target, *this); }

  /// Return a signed integral vector containing -1 for true elements and 0 otherwise.
  constexpr typename traits::signed_vec_for_mask operator-() const noexcept { return detail::mask_to_vec<-1, typename traits::element_signed_container_type>(target, *this); }

  /// Return the element-wise bitwise complement of the vector produced by unary plus.
  constexpr typename traits::signed_vec_for_mask operator~() const noexcept { return ~(+*this); }

  /// @} cpp26_mask_unary

  /// @addtogroup cpp26_mask_subscr
  /// @{

  /// @brief Select mask elements using an integral index vector.
  /// @param indexes The indexes selecting mask elements.
  /// @return A mask containing the selected elements.
  constexpr auto operator[](const vec_integral auto& indexes) const { return permute(*this, indexes); }

  /// @brief Return the mask element at the specified position.
  /// @param pos The element index, which must be less than size() and non-negative.
  constexpr bool operator[](simd_size_type pos) const& { return detail::subscript(target, *this, pos); }

  /// @} cpp26_mask_subscr

  /// @addtogroup cpp26_mask_binary
  /// @{

  /// Compute the element-wise logical AND.
  friend constexpr basic_mask operator&&(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_and{}); }

  /// Compute the element-wise logical OR.
  friend constexpr basic_mask operator||(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_or{}); }

  /// Compute the element-wise bitwise AND.
  friend constexpr basic_mask operator&(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_and{}); }

  /// Compute the element-wise bitwise OR.
  friend constexpr basic_mask operator|(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_or{}); }

  /// Compute the element-wise bitwise XOR.
  friend constexpr basic_mask operator^(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_xor{}); }

  /// @} cpp26_mask_binary

  /// @addtogroup cpp26_mask_cassign
  /// @{

  /// Compute the element-wise bitwise AND and assign the result.
  constexpr basic_mask& operator&=(const basic_mask& rhs) noexcept { *this = *this & rhs; return *this; }

  /// Compute the element-wise bitwise OR and assign the result.
  constexpr basic_mask& operator|=(const basic_mask& rhs) noexcept { *this = *this | rhs; return *this; }

  /// Compute the element-wise bitwise XOR and assign the result.
  constexpr basic_mask& operator^=(const basic_mask& rhs) noexcept { *this = *this ^ rhs; return *this; }

  /// @} cpp26_mask_cassign

  /// @addtogroup cpp26_mask_comparison
  /// @{

  /// Compare corresponding mask elements for equality.
  friend constexpr basic_mask operator==(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return !(lhs ^ rhs); }

  /// Compare corresponding mask elements for inequality.
  friend constexpr basic_mask operator!=(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return lhs ^ rhs; }

  /// Compare corresponding mask elements using less-than.
  friend constexpr basic_mask operator<(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return !lhs && rhs; }

  /// Compare corresponding mask elements using less-than-or-equal.
  friend constexpr basic_mask operator<=(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return !lhs || rhs; }

  /// Compare corresponding mask elements using greater-than.
  friend constexpr basic_mask operator>(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return lhs && !rhs; }

  /// Compare corresponding mask elements using greater-than-or-equal.
  friend constexpr basic_mask operator>=(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return lhs || !rhs; }

  /// @} cpp26_mask_comparison

  /// @addtogroup cpp26_mask_cond
  /// @{

  /// @brief Implementation of element-wise selection between two masks.
  /// @internal
  friend constexpr basic_mask simd_select_impl(const basic_mask& m, const basic_mask& if_true, const basic_mask& if_false) noexcept
    { return detail::select_if_else(target, m, if_true, if_false); }

  /// @brief Implementation of selection between two broadcast boolean values.
  /// @internal
  friend constexpr basic_mask simd_select_impl(const basic_mask& m, std::same_as<bool> auto if_true, std::same_as<bool> auto if_false) noexcept
    { return detail::select_if_else(target, m, basic_mask(if_true), basic_mask(if_false)); }

  /// @brief Implementation of element-wise vector selection using a mask.
  /// @internal
  template<vec_type _Tp>
  friend constexpr _Tp simd_select_impl(const basic_mask& m, const _Tp& if_true, const _Tp& if_false) noexcept
  requires (sizeof(_Tp) == _Bytes)
    { return detail::select_if_else(target, m, if_true, if_false); }

  /// @} cpp26_mask_cond

  /// @brief Output the mask to a stream as individual bits.
  /// @ingroup xvec_stream_output
  /// @param stream The destination stream.
  /// @param value The mask to output.
  /// @return The destination stream.
  friend std::ostream &operator<<(std::ostream &stream, const basic_mask& value) {
    stream << value.to_bitset();
    return stream;
  }
};

/// @addtogroup cpp26_mask_reductions
/// @{

/// @brief Return true if every mask element is true.
/// @param mask The mask to query.
/// @return True if all elements are true.
constexpr bool all_of(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    for (int i=0; i<mask.size(); ++i) if (!mask[i]) return false;
    return true;
  }
  else
    return detail::mask_all_of(target, mask);
}

/// @brief Apply all_of to one boolean element.
/// @param b The boolean element.
/// @return b.
constexpr bool all_of(std::same_as<bool> auto b) noexcept { return b; }

/// @brief Return true if any mask element is true.
/// @param mask The mask to query.
/// @return True if at least one element is true.
constexpr bool any_of(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    for (int i=0; i<mask.size(); ++i) if (mask[i]) return true;
    return false;
  }
  else
   return detail::mask_any_of(target, mask);
}

/// @brief Apply any_of to one boolean element.
/// @param b The boolean element.
/// @return b.
constexpr bool any_of(std::same_as<bool> auto b) noexcept { return b; }

/// @brief Return true if no mask element is true.
/// @param mask The mask to query.
/// @return True if all elements are false.
constexpr bool none_of(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    for (int i=0; i<mask.size(); ++i) if (mask[i]) return false;
    return true;
  }
  else
    return detail::mask_none_of(target, mask);
}

/// @brief Apply none_of to one boolean element.
/// @param b The boolean element.
/// @return !b.
constexpr bool none_of(std::same_as<bool> auto b) noexcept { return !b; }

/// @brief Count the true mask elements.
/// @param mask The mask to query.
/// @return The number of true elements.
constexpr simd_size_type reduce_count(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    simd_size_type count = 0;
    for (int i=0; i<mask.size(); ++i) if (mask[i]) count += 1;
    return count;
  }
  else
    return detail::mask_reduce_count(target, mask);
}

/// @brief Count one boolean element.
/// @param b The boolean element.
/// @return 1 if b is true, or 0 otherwise.
constexpr simd_size_type reduce_count(std::same_as<bool> auto b) noexcept { return +b; }

/// @brief Return the index of the first true mask element.
/// @param mask The mask to query.
/// @pre At least one mask element is true.
/// @return The lowest selected index.
constexpr simd_size_type reduce_min_index(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    simd_size_type i = 0;
    while ((i < mask.size()) && !mask[i]) i += 1;
    return i;
  }
  else
    return detail::mask_reduce_min_index(target, mask);
}

/// @brief Apply reduce_min_index to one boolean element.
/// @pre The input is true.
/// @return 0.
constexpr simd_size_type reduce_min_index(std::same_as<bool> auto) noexcept { return 0; }

/// @brief Return the index of the last true mask element.
/// @param mask The mask to query.
/// @pre At least one mask element is true.
/// @return The highest selected index.
constexpr simd_size_type reduce_max_index(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    simd_size_type i = mask.size() - 1;
    while (i >= 0 && !mask[i]) i -= 1;
    return i;
  }
  else
    return detail::mask_reduce_max_index(target, mask);
}

/// @brief Apply reduce_max_index to one boolean element.
/// @pre The input is true.
/// @return 0.
constexpr simd_size_type reduce_max_index(std::same_as<bool> auto) noexcept { return 0; }

/// @} cpp26_mask_reductions

/// @brief Generate a mask with its first n elements set to true.
/// @ingroup cpp29_p3440
/// @tparam _Vp The vector type whose mask is returned, or a scalar type for a boolean result.
/// @param n The number of elements to set. Counts larger than the mask width set every element.
/// @return A prefix mask for vector types, or n > 0 for scalar types.
template<typename _Vp>
constexpr auto mask_from_count(simd_size_type n) noexcept {
  if constexpr (vec_type<_Vp>) {
    using _Mp = typename _Vp::mask_type;
    return _Mp::__mask_from_count(n);
  }
  else
    return n > 0;
}

// CTAD, to convert a basic_mask into a basic_vec, equivalent to decltype(+mask).
/// @brief Deduce a vector type from a mask, equivalent to decltype(+k).
/// @ingroup cpp26_ctor
template<size_t _Bytes, typename _Abi>
  basic_vec(basic_mask<_Bytes, _Abi> k) ->
    basic_vec<typename basic_mask<_Bytes, _Abi>::traits::element_signed_container_type, _Abi>;


} // namespace _XVEC_NAMESPACE::simd