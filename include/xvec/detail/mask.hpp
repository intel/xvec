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

/// \defgroup simd_mask_cassign Mask compound assignment operators
/// @brief Compound mask assignment operations.

/// @defgroup simd_mask_binary Mask binary operators
/// @brief Apply an operator to each pair or respective bits from the input.

/// \defgroup simd_mask_reduce Mask reduction operations
/// @brief Reduce a simd mask to a single scalar value

/// @defgroup simd_mask_unary Mask unary operators
/// @brief Apply an operator to each bit in the input to generate a new bitwise mask

/// @defgroup simd_mask_constructor Mask constructors
/// @brief Construct simd masks

/// @brief A mask type
/// This allows the vector SIMD classes to handle all the required bit mask type
/// operations in a uniform manner without having to care about the underlying
/// mask type. The mask type can be one of two major styles:
///
///   Wide mask (e.g., Intel SSE, Intel AVX, Intel AVX2): A full vector of bits
///   with the same size as the SIMD class being used. Each element is all zeros
///   or all ones. Most x86 blend instructions use only the top bit, but having
///   a complete set of bits is useful for performing blend on other targets
///   using bitwise and/or/not/xor instructions.
///
///   Compact or narrow mask predicate (e.g., Intel AVX-512): A bitset
///   containing only the required number of bits for the number of elements in
///   the SIMD class being used. For example, __m256 treated as `float' elements
///   would have 8 bits. An ExtInt is used to allow the mask to be sized
///   arbitrarily.
///
/// This mask class hides these differences and allows masks to be converted for
/// use in blends, conditionals, selections, as well as operated on using and,
/// or, not, xor, and so on. For more advanced uses (e.g., popcount, any, none,
/// etc), the mask can be exported to a bitset, and bitset's own rich set of
/// operations used, instead of defining them all in this mask class.
template<std::size_t _Bytes, typename _Abi>
class basic_mask
{
public:

  using traits = detail::mask_traits<target_tag, _Bytes, _Abi::num_elements>;
  using builtin_type = typename traits::builtin_type;

  using abi_type = _Abi;

  using value_type = bool;

  using iterator = simd_iterator<basic_mask>;  ///< An iterator which allows a mask to be treated as a range.
  using const_iterator = simd_iterator<const basic_mask>;

  /// Return the number of elements contained in the mask
  static constexpr auto size = size_constant<_Abi::num_elements>{};

private:

  /// The builtin mask value.
  builtin_type mask;

public:

  constexpr basic_mask() = default;
  constexpr basic_mask(const basic_mask&) = default;
  constexpr basic_mask(basic_mask&&) noexcept = default;
  constexpr basic_mask& operator=(const basic_mask&) = default;
  constexpr basic_mask& operator=(basic_mask&&) noexcept = default;

  /// @brief Build a mask from a boolean value.
  /// @param b The boolean value to broadcast to all mask elements.
  constexpr explicit basic_mask(std::same_as<value_type> auto b) noexcept :
    mask([b] { if (b) return !basic_mask(); else return basic_mask(); }()) {}

  /// @brief Build a mask from a set of bits stored in an integral value.
  /// Each bit of the mask will be set or cleared as determined by the
  /// respective bit of the input integral. If the mask has a size bigger
  /// than the number of bits in the supplied integral value the remaining bits
  /// will be cleared.
  /// @ingroup simd_mask_constructor
  /// @param m The unsigned integral value from which to extract the bits into a mask.
  template <std::unsigned_integral T> requires (!std::same_as<T, value_type>)
  constexpr explicit basic_mask(T m) noexcept
    : basic_mask([m]() {
        if (std::is_constant_evaluated())
          return basic_mask([m](auto idx) { return (idx < std::numeric_limits<T>::digits) ? ((m >> idx) & 1) == 1 : 0; });
        else
          return detail::generate_mask_from_unsigned<basic_mask>(target, m);
    }()) {}

  /// @brief Create a mask from a mask representing a different vec type.
  /// The created mask is equivalent to the supplied input mask, but where
  /// the type of element being represented has been changed. Each bit of the
  /// new mask is set or cleared corresponding to the respective value of the
  /// supplied mask.
  /// @ingroup simd_mask_constructor
  /// @tparam _UBytes The number of bytes in the source mask
  /// @tparam _UAbi The ABI of the source mask elements
  /// @param m The input mask from which to create the new mask
  template<std::size_t _UBytes, typename _UAbi>
  requires (basic_mask<_UBytes, _UAbi>::size() == size())
  explicit constexpr basic_mask(const basic_mask<_UBytes, _UAbi>& m) : mask(detail::convert_mask<basic_mask>(target, m)) {}

  /// @brief Generate a mask using a generator to compute each bit.
  /// For each bit element the generator function is given the index of the
  /// element and will return the value (true or false) to be written into that
  /// mask element position. If the generator function is constexpr this can be used to
  /// construct pre-computed mask values.
  /// @ingroup simd_mask_constructor
  /// @param fn The generator function to compute each bit.
  constexpr basic_mask(std::invocable<simd_size_type> auto fn) : basic_mask(detail::generate_mask<basic_mask>(target, fn)) {}

  /// Generate a mask which represents the bottom N bits of the mask.
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

  /// @brief Allow a mask to be built from the corresponding bitset.
  /// The mask is created such that each bit is set or cleared
  /// corresponding to the value of the respective bit from the input bitset.
  /// @ingroup simd_mask_constructor
  /// @param b The bitset input value
  // :TODO: Not efficient if the bitset isn't constexpr. Need a better way to deal with that eventually.
  template<std::same_as<std::bitset<size()>> T>
  constexpr basic_mask(const T& b) noexcept : basic_mask([&](simd_size_type idx) { return b[idx]; }) {}

  /// Range interface.
  ///@{
  // missing begin for non-const
  constexpr iterator begin() noexcept { return {*this, 0}; }
  constexpr const_iterator begin() const noexcept { return {*this, 0}; }
  constexpr const_iterator cbegin() const noexcept { return {*this, 0}; }
  constexpr std::default_sentinel_t end() const noexcept { return {}; }
  constexpr std::default_sentinel_t cend() const noexcept { return {}; }
  ///@}

  /// Return the underlying mask value. This is explicit so that no silent conversions are
  /// permitted, which often break in unexpected ways. To avoid confusion, the user has to
  /// explicitly switch to the underlying type.
  // :TODO: Maybe to_builtin is sufficient?
  constexpr explicit operator builtin_type() { return mask; }

  /// @brief Convert a mask into a vec value where each element is set to 0 or
  /// 1 depending upon the state of its respective bit from the mask.
  /// @ingroup simd_constructor
  constexpr operator typename traits::signed_vec_for_mask() const { return detail::mask_to_vec<1, typename traits::element_signed_container_type>(target, *this); }

  /// Return the builtin mask value directly.
  constexpr auto to_builtin() const { return mask; }

  /// Construct a mask directly from the underlying builtin. This builtin might
  /// be a type which can be used for a standard constructor (e.g.,
  /// integral-like) so the `from_builtin` ensures that the desired effect is
  /// achieved directly.
  static constexpr basic_mask from_builtin(const auto& m) { basic_mask bm; bm.mask = traits::truncate_to_size(m); return bm; }

  /// Convert the mask into a register which can be passed to an intrinsic.
  constexpr auto to_register() const noexcept { return traits::to_register(mask); }

  /// Return a bitset representation of the values. std::bitset has a rich API and it makes more
  /// sense to use it directly rather than provide similar operations in mask instead (e.g.,
  /// avoid duplication, alternative implementations, etc.)
  constexpr std::bitset<size()> to_bitset() const noexcept { return detail::mask_to_bitset(target, *this); }

  /// Return an unsigned integral representation of the values using ullong. For
  /// masks which are bigger than a ullong, the upper bits must be zero.
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

  /// @{
  /// @ingroup simd_mask_unary
  /// @brief Construct a new mask by applying the unary operator to each bit of the input mask.
  constexpr basic_mask operator!() const noexcept { return detail::builtin_operator(target, *this, std::bit_not{}); }
  
  constexpr typename traits::signed_vec_for_mask operator+() const noexcept { return detail::mask_to_vec<1, typename traits::element_signed_container_type>(target, *this); }

  constexpr typename traits::signed_vec_for_mask operator-() const noexcept { return detail::mask_to_vec<-1, typename traits::element_signed_container_type>(target, *this); }

  constexpr typename traits::signed_vec_for_mask operator~() const noexcept { return ~(+*this); }
  /// @}

  /// @brief Permute a mask value by indexing it using the unsigned integral indexes supplied in another basic_vec value.
  /// @tparam _AbiU The index vec value ABI
  /// the permuted basic_vec value, and has the same number of elements as the index basic_vec.
  /// @tparam _Up The index type, which must be unsigned integral.
  /// @param indexes The indexes by which to permute
  /// @return The contents of this mask permuted by the indexes. The output of each element position is this[indexes[iota()]].
  constexpr auto operator[](const vec_integral auto& indexes) const { return permute(*this, indexes); }

  /// Return the value of a single element at the given index position.
  constexpr bool operator[](simd_size_type pos) const& { return detail::subscript(target, *this, pos); }

  ///@{
  /// @ingroup simd_mask_binary
  /// @param lhs The left hand value in the operation
  /// @param rhs The right hand value in the operation
  /// @return A mask value which records the result of applying the bit operator to respective bits of the input masks.
  friend constexpr basic_mask operator&&(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_and{}); }
  friend constexpr basic_mask operator||(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_or{}); }
  friend constexpr basic_mask operator&(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_and{}); }
  friend constexpr basic_mask operator|(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_or{}); }
  friend constexpr basic_mask operator^(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return detail::builtin_operator(target, lhs, rhs, std::bit_xor{}); }
  /// @}

  /// @{
  /// @ingroup simd_mask_cassign
  /// @param rhs The right hand value of the assignment operator
  /// @return The bitwise operation value
  constexpr basic_mask& operator&=(const basic_mask& rhs) noexcept { *this = *this & rhs; return *this; }
  constexpr basic_mask& operator|=(const basic_mask& rhs) noexcept { *this = *this | rhs; return *this; }
  constexpr basic_mask& operator^=(const basic_mask& rhs) noexcept { *this = *this ^ rhs; return *this; }
  /// @}

  /// @{
  /// @brief Compare two mask values.
  /// @ingroup simd_mask_compare
  /// @param lhs The left hand operand
  /// @param rhs the right hand operand
  /// @return A new mask in which every output mask bit is set to the result of
  /// the comparison of respective input mask bits.
  friend constexpr basic_mask operator==(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return !(lhs ^ rhs); }
  friend constexpr basic_mask operator!=(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return lhs ^ rhs; }
  friend constexpr basic_mask operator<(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return !lhs && rhs; }
  friend constexpr basic_mask operator<=(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return !lhs || rhs; }
  friend constexpr basic_mask operator>(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return lhs && !rhs; }
  friend constexpr basic_mask operator>=(const basic_mask& lhs, const basic_mask& rhs) noexcept
    { return lhs || !rhs; }
  /// @}

  ///@{
  /// @brief Given two masks, create a new mask which blends respective bits
  /// from two inputs according to some selector mask. select works like
  /// (is_bit_set ? a : b) for each element.
  /// @ingroup simd_mask_cond
  friend constexpr basic_mask simd_select_impl(const basic_mask& m, const basic_mask& if_true, const basic_mask& if_false) noexcept
    { return detail::select_if_else(target, m, if_true, if_false); }
  friend constexpr basic_mask simd_select_impl(const basic_mask& m, std::same_as<bool> auto if_true, std::same_as<bool> auto if_false) noexcept
    { return detail::select_if_else(target, m, basic_mask(if_true), basic_mask(if_false)); }
  template<vec_type _Tp>
  friend constexpr _Tp simd_select_impl(const basic_mask& m, const _Tp& if_true, const _Tp& if_false) noexcept
  requires (sizeof(_Tp) == _Bytes)
    { return detail::select_if_else(target, m, if_true, if_false); }
  ///@}

  /// @brief Output the mask to a stream as individual bits.
  /// @ingroup simd_stream_output
  friend std::ostream &operator<<(std::ostream &stream, const basic_mask& value) {
    stream << value.to_bitset();
    return stream;
  }
};

/// @brief Return true if every bit in the mask is set. Note that a boolean value is treated as a mask with
/// a single element.
/// @ingroup simd_mask_reduce
/// @param mask The mask to query
/// @return True if all the bits in the mask are set, false otherwise.
///@{
constexpr bool all_of(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    for (int i=0; i<mask.size(); ++i) if (!mask[i]) return false;
    return true;
  }
  else
    return detail::mask_all_of(target, mask);
}

constexpr bool all_of(std::same_as<bool> auto b) noexcept { return b; }
///@}

/// @brief Return true if any bit in the mask is set. Note that a boolean value is treated as a mask with
/// a single element.
/// @ingroup simd_mask_reduce
/// @param mask The mask to query
/// @return True if any bit in the mask is set.
///@{
constexpr bool any_of(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    for (int i=0; i<mask.size(); ++i) if (mask[i]) return true;
    return false;
  }
  else
   return detail::mask_any_of(target, mask);
}

constexpr bool any_of(std::same_as<bool> auto b) noexcept { return b; }
///@}

/// @brief Return true if no bit in the mask is set. Note that a boolean value is treated as a mask with
/// a single element.
/// @ingroup simd_mask_reduce
/// @param mask The mask to query
/// @return True if no bit in the mask is set.
///@{
constexpr bool none_of(const mask_type auto& mask) noexcept {
  if (std::is_constant_evaluated())
  {
    for (int i=0; i<mask.size(); ++i) if (mask[i]) return false;
    return true;
  }
  else
    return detail::mask_none_of(target, mask);
}

constexpr bool none_of(std::same_as<bool> auto b) noexcept { return !b; }
///@}

/// @brief Return the number of mask bits which are set. Note that a boolean value is treated as a mask with
/// a single element.
/// @ingroup simd_mask_reduce
/// @param mask The mask to query
/// @return The total count of set bits in the mask input.
///@{
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

constexpr simd_size_type reduce_count(std::same_as<bool> auto b) noexcept { return +b; }
///@}

/// @brief Return the index of the lowest set bit. Note that a boolean value is treated as a mask with
/// a single element.
/// @ingroup simd_mask_reduce
/// @param mask The mask to query
/// @return The index of the lowest set bit.
///@{
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

constexpr simd_size_type reduce_min_index(std::same_as<bool> auto) noexcept { return 0; }
///@}

/// @brief Return the index of the highest set bit. Note that a boolean value is treated as a mask with
/// a single element.
/// @ingroup simd_mask_reduce
/// @param mask The mask to query
/// @return The index of the highest set bit.
///@{
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

constexpr simd_size_type reduce_max_index(std::same_as<bool> auto) noexcept { return 0; }
///@}

/// @brief Generate a mask with the bottom N bits set. @tparam _Vp The type of
/// the mask to generate, which determines the number of bits in the mask.
/// @param n The number of bits to set in the output mask. If n is bigger than
/// the number of bits in the mask, all bits will be set.
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
template<size_t _Bytes, typename _Abi>
  basic_vec(basic_mask<_Bytes, _Abi> k) ->
    basic_vec<typename basic_mask<_Bytes, _Abi>::traits::element_signed_container_type, _Abi>;


} // namespace _XVEC_NAMESPACE::simd
