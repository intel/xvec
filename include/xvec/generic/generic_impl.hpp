//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <bitset>
#include <cmath> // std::ceil, std::floor, std::trunc, std::round, std::rint, std::sqrt
#include <cstddef> // std::size_t
#include <cstdint> // std::uint8_t, std::uint64_t
#include <limits> // std::numeric_limits
#include <memory> // std::to_address
#include <type_traits>
#include <algorithm> // std::min, std::max
#include <utility> // std::index_sequence
#include <cstring> // for std::memcpy

#include <xvec/detail/config.hpp>
#include <xvec/detail/utilities.hpp>

namespace _XVEC_NAMESPACE::simd
{

inline constexpr simd_size_type zero_element = 9999123;   // No specific value other than being big enough to fall outside valid vec indexes.
inline constexpr simd_size_type uninit_element = 9999124; // No specific value other than being big enough to fall outside valid vec indexes.

/// @brief Generate a variable containing the iota sequence (i.e., 0, 1, 2, ...
/// size-1) for the given simd::vec type. This can be used as the basis of generating
/// other sequences, including those used in constexpr scenarios. For example,
/// to generate all multiples of 3 you can do `iota<T> * 3'. Using the constexpr
/// nature of iota, along with the constexpr operations and functions makes it
/// natural to build simd::vec constants with particular properties. Perhaps even
/// easier than using the generator constructor. @ingroup simd_constructor
/// @tparam _Tp The vec or arithmetic type for which to generate the iota sequence
/// @return A simd::vec or scalar value containing the iota sequence for the type
template<typename _Tp>
  requires (vec_type<_Tp> || std::is_arithmetic_v<_Tp>)
inline constexpr _Tp iota =
  []{ if constexpr (vec_type<_Tp>) return _Tp([](simd_size_type i) { return typename _Tp::value_type(i); });
      else return _Tp{}; }();

namespace detail
{

// :TODO: Move into utilities.
const auto perm_uninitResize = [](auto idx, auto isize) { return idx < isize ? idx : uninit_element; };

// Provide a way to initialise the values in a vec_impl::vector from a generator.
template<vec_type _Vp, typename _Gp>
constexpr auto generate(_Gp generator) {
  if constexpr (complex_number<typename _Vp::value_type>)
  {
    using _Tp = typename _Vp::traits::element_type;
    return _Vp([=]<std::size_t... _Idx>(std::index_sequence<_Idx...>)  {
      return typename _Vp::traits::builtin_type{_Tp((_Idx % 2) ? generator(_Idx / 2).imag() : generator(_Idx / 2).real())...};
    }(std::make_index_sequence<_Vp::size() * 2>()));
  }
  else
  {
    return _Vp([=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
      return typename _Vp::traits::builtin_type{std::bit_cast<typename _Vp::traits::element_type>(generator(_Idx))...};
    }(std::make_index_sequence<_Vp::size()>()));
  }
}

template<simd_size_type _NewSize, vec_or_mask_type _Vp>
constexpr resize_t<_NewSize, _Vp> fit_to_size(const _Vp& v) { 
  if constexpr (_NewSize == _Vp::size()) return v;
  else return permute<_NewSize>(v, [](simd_size_type i) { return (i < _Vp::size()) ? i : zero_element; });
}

template<vec_type _Vp, contiguous_sized_range _Range, typename _Flags>
constexpr void mandates_for_store(const _Vp&, const _Range&, _Flags f)
{
  using _Tp = typename _Vp::value_type;
  using _Rp = std::ranges::range_value_t<_Range>;

  static_assert(detail::contains_flag<detail::flag_convert>(f) ||
                !detail::needs_explicit_conversion<_Tp, _Rp>,
                "Store or scatter conversion requires flag_convert");
}

template<vec_type _Vp, contiguous_sized_range _Range, typename _Flags>
constexpr void mandates_for_load(const _Range&, const _Vp&, _Flags f)
{
  using _Tp = typename _Vp::value_type;
  using _Rp = std::ranges::range_value_t<_Range>;

  static_assert(detail::vectorizable<_Rp>);
  static_assert(detail::explicitly_convertible_to<_Rp, _Tp>);
  static_assert(std::same_as<std::remove_cvref_t<_Vp>, _Vp>);

  // flag_convert is required if and only if the equivalent simd converting
  // constructor would be explicit.
  static_assert(detail::contains_flag<detail::flag_convert>(f) ||
                !detail::needs_explicit_conversion<_Rp, _Tp>,
                "Load or gather conversion requires flag_convert");
}

/// Convert a vec_type into the appropriate container type. Each element is the
/// same size as the original and the number of elements is the same. The return
/// value is bitwise equivalent to the original, but with a new type. This
/// function is useful for converting a simd of a specific element type into a
/// generic simd container which can be manipulated as a bucket-of-bits. It can
/// also be used to convert a mask into a suitable container, allowing generic
/// permute-like operations to work on masks too.
///@{
template<vec_type _Vp>
constexpr auto vec_as_container(const _Vp& v) { return simd_bit_cast<typename _Vp::traits::element_container_type>(v); }

/// The mask equivalent uses the negate operation to generate the implementation builtin mask
template<mask_type _Mp> constexpr auto vec_as_container(const _Mp& m) { return vec_as_container(-m); }
///@}

/// Access the index at the given index of the basic_vec or basic_mask.
/// \param pos The position of the accessed element.
///@{
template<vec_or_mask_type _Vp>
constexpr typename _Vp::value_type subscript(generic_tag, const _Vp& v, simd_size_type pos) {
  assert(pos < _Vp::size());
  auto t = detail::vec_as_container(v).to_builtin()[pos];

  if constexpr (vec_complex<_Vp>)
  {
    // std::bit_cast for std::complex in clang doesn't work until 2025.1.0.
    // Before that version it is necessary to get the complex value from its
    // individual components.
    using _E = complex_element_type<typename _Vp::value_type>;
    auto asCmplxArray = std::bit_cast<std::array<_E, 2>>(t);
    return std::complex<_E>(asCmplxArray[0], asCmplxArray[1]);
  }
  else if constexpr (mask_type<_Vp>)
    return typename _Vp::value_type(t);
  else
    return std::bit_cast<typename _Vp::value_type>(t);
}

template<mask_type _Mp>
constexpr bool subscript(compact_mask_tag, const _Mp& v, simd_size_type pos)
  { return (v.to_register() >> pos) & 1; }
///@}

/// Construct a mask from the given generator.
///@{
template<mask_type _Mp, typename _Gp>
constexpr _Mp generate_mask(generic_tag, _Gp generator) {
  using _Tp = container_for_num_bytes<simd_mask_element_size_v<_Mp>>;
  using _Vp = typename _Mp::builtin_type;

  auto r = [=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
    return _Vp{_Tp(generator(size_constant<_Idx>()) ? ~_Tp() : _Tp())...};
  }(std::make_index_sequence<_Mp::size()>());

  return _Mp::from_builtin(r);
}

template<mask_type _Mp, typename _Gp>
constexpr _Mp generate_mask(compact_mask_tag, _Gp generator) {
  return _Mp::from_builtin([=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
    return ((typename _Mp::builtin_type(generator(size_constant<_Idx>{})) << _Idx) | ...);
  } (std::make_index_sequence<_Mp::size>()));
}
///@}

template<mask_type _Mp>
constexpr std::bitset<_Mp::size>
mask_to_bitset(generic_tag, const _Mp& mask) noexcept
{
  // Very inefficient, but all targets should provide their own version. Note
  // also that bitsets are not constexpr before C++23.
  std::bitset<_Mp::size> result;
  for (int i=0; i<_Mp::size; i++)
    result[i] = (mask[i] != 0);
  return result;
}

constexpr unsigned long long mask_to_ullong(generic_tag, const mask_type auto& mask)
  { return mask.to_bitset().to_ullong(); }

constexpr unsigned long long mask_to_ullong(compact_mask_tag, const mask_type auto& mask)
  { return (unsigned long long)mask.to_builtin(); }

/// @brief Permute a set of vec elements by the indexes supplied in another vec. The
/// source data is forced into memory to turn this into whatever memory-gather
/// operation is efficiently supported by the target, as that is likely to be
/// faster than trying to read a series of elements from run-time positions in a
/// register.
/// @tparam _Vp The vec type
/// @tparam _Ip The index vec type
/// @param v The vec value to permute
/// @param indexes The vec of indexes
/// @return The permuted vec value
///@{
template<vec_type _Vp, vec_integral _Ip>
constexpr resize_t<_Ip::size(), _Vp>
permute(generic_tag, const _Vp& v, const _Ip& indexes)
  { return partial_gather_from(std::span(reinterpret_cast<const typename _Vp::value_type*>(&v), _Vp::size()), indexes); }

template<mask_type _Mp, vec_integral _Ip>
constexpr resize_t<_Ip::size(), _Mp>
permute(generic_tag, const _Mp& m, const _Ip& indexes)
  { return resize_t<_Ip::size(), _Mp>::from_builtin(permute(vec_as_container(m), indexes).to_builtin()); }

template<mask_type _Mp, vec_integral _Ip>
constexpr resize_t<_Ip::size(), _Mp>
permute(compact_mask_tag, const _Mp& m, const _Ip& indexes)
  {  return permute(-m, indexes) != cw<0>; }
///@}

/// Choose one or the other element, depending upon the value of a mask.
/// @tparam _Vp The vec type
/// @param mask The mask to use for selection
/// @param if_true The value to use if the mask bit is set
/// @param if_false The value to use if the mask bit is clear
/// @return The selected simd value
///@{
template<vec_type _Vp>
constexpr _Vp
select_if_else(generic_tag, const typename _Vp::mask_type& mask, const _Vp& if_true, const _Vp& if_false)
{
  using _C = typename _Vp::traits::element_container_type;

  auto tm = mask.to_builtin();
  auto ta = simd_bit_cast<_C>(if_true).to_builtin();
  auto tb = simd_bit_cast<_C>(if_false).to_builtin();

  rebind_t<_C, _Vp> r;
  if (std::is_constant_evaluated())
    r = (tm & ta) | ((~tm) & tb); // operator? doesn't work while in constexpr.
  else
    r = tm ? ta : tb;

  return simd_bit_cast<typename _Vp::value_type>(r);
}

template<vec_type _Vp>
constexpr _Vp
select_if_else(compact_mask_tag, const typename _Vp::mask_type& mask, const _Vp& if_true, const _Vp& if_false)
  { return _Vp([=](auto i) { return mask[i] ? if_true[i] : if_false[i]; }); }

template<mask_type _Mp>
constexpr _Mp
select_if_else(compact_mask_tag, const _Mp& if_mask, const _Mp& if_true, const _Mp& if_false) {
  auto r = (if_true.to_builtin() & if_mask.to_builtin()) | (if_false.to_builtin() & ~if_mask.to_builtin());
  return _Mp::from_builtin(r);
}

/// @brief Select between two simd masks based on a mask of the same type.
/// @tparam _Mp The mask type.
/// @param mask The mask to use for selection
/// @param if_true The value to use if the mask bit is set
/// @param if_false The value to use if the mask bit is clear
/// @return The selected simd mask
 template<mask_type _Mp>
constexpr _Mp
select_if_else(generic_tag, const _Mp& mask, const _Mp& if_true, const _Mp& if_false) noexcept
{ return _Mp::from_builtin(mask.to_builtin() ? if_true.to_builtin() : if_false.to_builtin()); }

/// @brief Implement generic stores.
/// @tparam _Vp the type of the source.
/// @tparam _Up The element type for the vec being stored.
/// @tparam _Extent The extent of the output span.
/// @tparam _Flags Memory flags. This can be used to enable or disable boundary checking.
/// @param value_original The vec value to write to memory.
/// @param to The output buffer as a span.
/// @param mask_original The mask value controlling which vec elements will be written to memory.
/// @param flags Memory flags. This can be used to enable or disable boundary checking.
///@{
template<vec_type _Vp, typename _Up, std::size_t _Extent, typename _Flags>
constexpr void store_masked(generic_tag, const _Vp& value_original, std::span<_Up, _Extent> to,
                            const typename _Vp::mask_type& mask_original, _Flags flags)
{
  mandates_for_store(value_original, to, flags);

  auto limit = _Vp::size();
  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds<rebind_t<_Up, _Vp>>("store", to, mask_original);
  else
    limit = std::min<std::size_t>(to.size(), _Vp::size());

  for (int i=0; i<limit; i++)
    if (mask_original[i])
      to[i] = static_cast<_Up>(value_original[i]);
}

template<vec_type _Vp, typename _Up, std::size_t _Extent, typename _Flags>
constexpr void store(generic_tag, const _Vp& value_original, std::span<_Up, _Extent> to, _Flags flags)
{
  mandates_for_store(value_original, to, flags);

  auto limit = _Vp::size();
  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds<rebind_t<_Up, _Vp>>("store", to);
  else
      limit = std::min<std::size_t>(to.size(), _Vp::size());

  for (int i=0; i<limit; i++)
    to[i] = static_cast<_Up>(value_original[i]);
}
///@}

/// @brief Load a basic_vec from the given memory span, returning a simd with the given type as _V.
/// @tparam _V The simd type to load
/// @tparam _Up The type of the underlying memory
/// @tparam _Extent The extent of the span
/// @tparam _Flags The type of the flags
/// @param from The memory span from which to perform the load
/// @param flags The flags controlling the load
/// @return The simd loaded from memory
template<vec_type _V, typename _Up, std::size_t _Extent, typename _Flags>
constexpr _V load(generic_tag, std::span<_Up, _Extent> from, _Flags flags)
{
  mandates_for_load(from, _V{}, flags);

  using SourceType = rebind_t<_Up, _V>;

  // :TODO: Make use of the fixed size where permitted.
  std::remove_const_t<typename SourceType::builtin_type> result = {};
  if constexpr (contains_flag<flag_unchecked>(flags))
  {
    checkStaticMemoryBounds<SourceType>("load", from);
    if (std::is_constant_evaluated())
      return _V([=](auto i) -> typename _V::value_type { return from[i]; });
    else
    {
      memcpy(&result, from.data(), sizeof(_Up) * _V::size);
      return _V(SourceType(result));
    }
  }
  else
  {
    // Use a runtime check to ensure no overrun for smaller sources.
    auto numElementsToLoad = std::min<std::size_t>(_V::size, from.size());

    if (std::is_constant_evaluated())
      return _V([=](auto i) -> typename _V::value_type { return i < numElementsToLoad ? from[i] : _Up(); });
    else
    {
      __builtin_memcpy(&result, from.data(), sizeof(_Up) * numElementsToLoad);
      return _V(SourceType(result));
    }
  }

}

/// @brief Load a basic_vec from the given memory span, using a mask.
/// @tparam _V The simd type to load
/// @tparam _Up The type of the underlying memory
/// @tparam _Extent The extent of the span
/// @tparam _Flags The type of the flags
/// @param from The memory span from which to perform the load
/// @param m The mask to use for loading
/// @param flags The flags controlling the load
/// @return The simd loaded from memory, with masked elements zeroed
template<vec_type _V, typename _Up, std::size_t _Extent, typename _Flags>
constexpr _V load_masked(generic_tag, std::span<_Up, _Extent> from, const typename _V::mask_type& m, _Flags flags)
{
  mandates_for_load(from, _V{}, flags);

  using SourceType = vec<_Up, _V::size>;

  std::remove_const_t<typename SourceType::builtin_type> result = {};
  if constexpr (contains_flag<flag_unchecked>(flags))
  {
    checkStaticMemoryBounds<SourceType>("load", from, m);
    if (std::is_constant_evaluated())
      return select(m, _V([=](auto i) -> typename _V::value_type { return from[i]; }), _V());
    else
    {
      memcpy(&result, from.data(), sizeof(_Up) * _V::size);
      return select(m, _V(SourceType(result)), _V());
    }
  }
  else
  {
    // Use a runtime check to ensure no overrun for smaller sources.
    auto numElementsToLoad = std::min<std::size_t>(_V::size, from.size());
    if (std::is_constant_evaluated())
      return select(m, _V([=](auto i) -> typename _V::value_type { return i < numElementsToLoad ? from[i] : _Up(); }), _V());
    else
    {
      memcpy(&result, from.data(), sizeof(_Up) * numElementsToLoad);
      return select(m, _V(SourceType(result)), _V());
    }
  }

}

/// Wrapper around the builtin_shufflevector function, enabling it to be called
/// for vec<> types. This function also handles some of the differences in
/// storage layout between different compilers.
template<simd_size_type _OutSize, vec_or_mask_type _Vp, typename _Gp>
constexpr resize_t<_OutSize, _Vp> permute_pair(const _Vp& v0, const _Vp& v1, _Gp) {
  auto c0 = vec_as_container(v0);
  auto c1 = vec_as_container(v1);

  using _C = typename decltype(c0)::value_type;
  constexpr int _Np = _Vp::size;

  // Tweak the numbers to work on gcc and clang uniformly.
  auto getIndex = [](auto idx) {
    // Compute the required index to use. Note that this will be called for
    // every storage element, but any elements past the end of the actual
    // output storage will be left as undefined. This is required since gcc always
    // initialises the entire simd::vec, but the generator function shouldn't be
    // called for values which don't map to actual elements.
    if constexpr (idx >= _OutSize)
      return size_constant<-1>();
    else
    {
      constexpr int outIdx = simd_size_type(_Gp{}(idx));

      // Adjust the index to reference the second vector where necessary. Clang
      // assumes that two vectors of size N will have indexes [0..2N), but gcc may
      // insert padding.
      constexpr int paddingAdjustedIdx = outIdx < _Np ? outIdx : int(outIdx - _Np) + int(vec_traits<target_tag, _C, _Np>::_numStorageElements);

      return size_constant<paddingAdjustedIdx>();
    }
  };

  using OutTraits = vec_traits<target_tag, _C, _OutSize>;
  auto r = [=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) -> typename OutTraits::builtin_type {
    return __builtin_shufflevector(c0.to_builtin(), c1.to_builtin(), (getIndex(std::integral_constant<int, _Idx>()).value)...);
  }(std::make_index_sequence<OutTraits::_numStorageElements>());

  if constexpr(vec_type<_Vp>)
    return simd_bit_cast<typename _Vp::value_type>(vec<_C, _OutSize>(r));
  else
    // Mask type goes directly from container to mask without a need to bit cast.
    return resize_t<_OutSize, _Vp>::from_builtin(r);
}

/// @brief Permute a vec values using the indexes created by the
/// generator function.
/// @tparam _OutSize The output simd size
/// @tparam _Gp The generator function type
/// @tparam _Vp The type of vec to permute
/// @param generator The generator function
/// @param v The vec value to permute.
/// @return The permuted vec value.

template<simd_size_type _OutSize, vec_or_mask_type _Vp, typename _Gp>
constexpr resize_t<_OutSize, _Vp>
generated_permute(generic_tag, const _Vp& v, _Gp) {
  constexpr simd_size_type _Np = _Vp::size;

  auto c0 = vec_as_container(v);
  using _C = typename decltype(c0)::value_type;

  // The second value in the shufflevector arguments is zero, so return the
  // first index of that as the zero index. Note that GCC stores values
  // differently to clang (power-of-2) so return the actual index of the
  // storage, not the apparent type.
  constexpr auto zeroIndex = vec_traits<target_tag, _C, _Np>::_numStorageElements;

  auto getIndex = []<auto idx>() {
    // Do not invoke the generator for padding elements in the output storage.
    if constexpr (idx >= _OutSize) {
      return size_constant<-1>{};
    } else {
      // Get the generated index, passing the input size when supported.
      constexpr simd_size_type gi = [] {
        if constexpr (index_generator_function_with_size<_Gp>)
          return _Gp{}(size_constant<idx>{}, _Np);
        else
          return _Gp{}(size_constant<idx>{});
      }();

      if constexpr (gi == zero_element)
        return size_constant<zeroIndex>{};
      else if constexpr (gi == uninit_element)
        // Mark as unknown
        return size_constant<-1>{};
      else
      {
        static_assert(gi >= 0 && gi < _Np, "Computed index isn't in valid input range");
        return size_constant<gi>{};
      }
    }
  };

  using OutTraits = vec_traits<target_tag, _C, _OutSize>;

  typename OutTraits::builtin_type r;
  constexpr auto emptyBuiltin = decltype(c0.to_builtin()){};
  if (std::is_constant_evaluated())
  {
    // Nasty code to work around differences in gcc and llvm. In a constexpr
    // context gcc is happy with an index of -1 (meaning uninitialised), but
    // clang is not. But putting a work-around inside the lambda to enable clang
    // to detect and remove -1 in a constexpr context then breaks gcc's
    // constexpr evaluation. The `fix` is to detect constexpr context outside
    // the loop, so that the inner-most part of the shufflevector doesn't have
    // to deal with is_constnt_evaulated directly, but the downside is two
    // slightly different copies of the lambda.
    r = [=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) -> typename OutTraits::builtin_type {
      auto fixUninit = [](auto idx) { return idx < 0 ? zeroIndex : idx; };
      return __builtin_shufflevector(c0.to_builtin(), emptyBuiltin, fixUninit(getIndex.template operator()<_Idx>().value)...);
    }(std::make_index_sequence<OutTraits::_numStorageElements>());
  } else {
    // As above, but without the -1 workaround.
    r = [=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) -> typename OutTraits::builtin_type {
      return __builtin_shufflevector(c0.to_builtin(), emptyBuiltin, (getIndex.template operator()<_Idx>().value)...);
    }(std::make_index_sequence<OutTraits::_numStorageElements>()); 
  }

  if constexpr (vec_type<_Vp>)
    return simd_bit_cast<typename _Vp::value_type>(vec<_C, _OutSize>(r));
  else
    // Mask type goes directly from container to mask without a bit cast.
    return resize_t<_OutSize, _Vp>::from_builtin(r);
}

template<simd_size_type _OutSize, mask_type _Mp, typename _Gp>
constexpr auto
generated_permute(compact_mask_tag, const _Mp& m, _Gp)
  { return permute<_OutSize>(-m, _Gp{}) != cw<0>; }

/// Permute a set of simd elements by the supplied mask. This is a generic
/// version which should be overridden by the target if a more efficient variant
/// exists. Note that the output simd will have the same number of elements as
/// the mask used in the selection, and the selection mask must be no bigger
/// than the available input simd.
/// @tparam _Vp The vec type
/// @param v The simd value to expand
/// @param mask The mask to use for expansion
/// @param original The original vec value
/// @return The expanded vec value
template<vec_type _Vp>
constexpr _Vp
expand_by_mask(generic_tag, const _Vp& v, const typename _Vp::mask_type& mask, const _Vp& original)
{
  typename _Vp::value_type result[_Vp::size];
  unchecked_store(original, result);

  auto inIter = (const typename _Vp::value_type*)(&v);

  for (int i=0; i<_Vp::size; ++i)
  {
    if (mask[i])
    {
      result[i] = *inIter;
      inIter++;
    }
  }

  return _Vp(result);
}

/// Permute a set of simd elements representing a mask using the other supplied
/// mask. This is a generic version which should be overridden by the target if a
/// more efficient variant exists.
///@{
template<mask_type _Mp>
constexpr _Mp
expand_mask_by_mask(generic_tag, const _Mp& value, const _Mp& mask, const _Mp& original = {})
  { return _Mp::from_builtin(expand(vec_as_container(value), mask, vec_as_container(original)).to_builtin()); }

template<mask_type _Mp>
constexpr _Mp
expand_mask_by_mask(compact_mask_tag, const _Mp& v, const _Mp& selector, const _Mp& original = {}) 
  { return expand(-v, selector, -original) != cw<0>; }
///@}

/// Special type used to indicate that the compress operation will not fill unused values.
struct compress_fill_uninitialized_t {};

/// Compress a set of simd elements by the supplied mask. This is a generic
/// version which should be overridden by the target if a more efficient variant
/// exists.
/// @tparam _Vp The vec type
/// @tparam _FillType The type of the fill value
/// @param v The simd value to compress
/// @param mask The mask to use for compression
/// @param fill_value The value to use for unfilled elements (optional)
/// @return The compressed simd value
template<vec_type _Vp, typename _FillType = compress_fill_uninitialized_t>
constexpr _Vp
compress_by_mask(generic_tag, const _Vp& v, const typename _Vp::mask_type& mask, _FillType fill_value = {})
{
  constexpr int _Np = _Vp::size;
  typename _Vp::value_type result[_Np];  // The output is deliberately left undefined to avoid users thinking that they can rely on
                                          // contents of the result which are not explicitly written.

  // If a fill value is supplied then use that. Typically, filling in unused
  // values costs cycles so prefer to avoid this is possible.
  if constexpr (!std::same_as<compress_fill_uninitialized_t, _FillType>)
    unchecked_store(_Vp(fill_value), result);

  auto outIter = result; 

  auto inIter = (const typename _Vp::value_type*)(&v);
  auto maskIter = (const typename _Vp::value_type*)(&mask);

  for (int i=0; i<_Np; ++i)
  {
    if (maskIter[i])
    {
      *outIter = inIter[i];
      outIter++;
    }
  }

  return _Vp(result);
}

template<vec_type _V>
constexpr _V
compress_by_mask(compact_mask_tag, const _V& values, const typename _V::mask_type& selector,
                 typename _V::value_type fill_value = {})
{
  std::array<typename _V::value_type, _V::size> result;
  std::fill(result.begin(), result.end(), fill_value);

  auto outIdx = 0;
  for (int i=0; i<_V::size(); ++i)
  {
    if (selector[i])
    {
      result[outIdx] = values[i];
      outIdx++;
    }
  }

  return _V(result);
}

/// Permute a set of simd elements representing a mask using the other supplied
/// mask. This is a generic version which should be overridden by the target if a
/// more efficient variant exists.
///@{
template<mask_type _Mp>
constexpr _Mp
compress_mask_by_mask(generic_tag, const _Mp& v, const _Mp& mask)
  { return _Mp::from_builtin(compress(vec_as_container(v), mask).to_builtin()); }

template<mask_type _Mp>
constexpr _Mp
compress_mask_by_mask(generic_tag, const _Mp& v, const _Mp& mask, bool fill_value)
{
  using FillType = typename _Mp::traits::element_container_type;
  FillType fill = fill_value ? ~FillType() : FillType();
  return _Mp::from_builtin(compress(vec_as_container(v), mask, fill).to_builtin());
}

template<mask_type _M>
constexpr _M
compress_mask_by_mask(compact_mask_tag, const _M& v, const _M& selector) {
  // Not an efficient implementation, so targets with compact masks should provide their own version of this function.
  return compress(-v, selector) != cw<0>;
}

template<mask_type _M>
constexpr _M
compress_mask_by_mask(compact_mask_tag, const _M& v, const _M& selector, bool fill_value)
{
  auto allFill = _M(fill_value);

  // Handling the fill is done by exploiting the knowledge that the normal
  // compress will zero out the excess bits. If the fill value must be
  // true, then the excess bits need to be set which in turn means we want to
  // zero fill and then flip the bits. Consequently the input bits must also be
  // flipped, so that they can be flipped again to set them. An example
  // illustrates this:
  //
  //  mask:           101010
  //  input:          110010
  //  flipped Input:  001101
  //  compressed:     000011
  //  flipped output: 111100
  //
  // The input and final output only need to be flipped on a true fill value, so
  // use an xor toggle.
  //
  // This code is very efficient because it only requires an extra two very
  // cheap XOR operations over the non-fill version.
  return compress(v ^ allFill, selector) ^ allFill;
}
///@}

/// Gather the contents of memory from the supplied index positions.
// :TODO: :COMPILER: Expose the gather llvm primitive and use it directly?
/// @tparam _Range The type of the range
/// @tparam _Idx The index type
/// @tparam _IdxAbi The ABI of the index simd::vec
/// @tparam _Flags The type of the flags
/// @param r The range from which to gather
/// @param indexes The simd of indexes
/// @param flags The flags controlling the gather
/// @return The gathered simd value
template<detail::contiguous_sized_range _Range, std::integral _Idx, typename _IdxAbi, typename... _Flags>
constexpr auto gather_from(generic_tag, const _Range& r, const basic_vec<_Idx, _IdxAbi>& indexes, flags<_Flags...> flags)
{
  using _Tp = std::ranges::range_value_t<_Range>;
  const auto rmax = r.size();

  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds("gather", indexes, rmax);

  constexpr int numIndexes = basic_vec<_Idx, _IdxAbi>::size();

  // Put the indexes and the result into memory, as this generates better code
  // than trying to pull out individual elements of a simd register to pass to a
  // separate load instruction, and to put the results back into a register.
  const auto indexPtr = reinterpret_cast<const _Idx*>(&indexes);
  _Tp result[numIndexes];

  for (int i=0; i<numIndexes; ++i)
  {
    auto idx = indexPtr[i];
    result[i] = contains_flag<flag_unchecked>(flags) || (idx >= 0 && std::cmp_less(+idx, rmax)) ? r[idx] : _Tp();
  }

  return vec<_Tp, numIndexes>(result);
}

/// Gather the contents of memory from the supplied index positions, using a mask.
/// @tparam _Range The type of the range
/// @tparam _Idx The index type
/// @tparam _IdxAbi The ABI of the index simd::vec
/// @tparam _Flags The type of the flags
/// @param r The range from which to gather
/// @param mask The mask to use for gathering
/// @param indexes The simd of indexes
/// @param flags The flags controlling the gather
/// @return The gathered simd value
template<detail::contiguous_sized_range _Range, std::integral _Idx, typename _IdxAbi, typename... _Flags>
constexpr auto gather_from(generic_tag, const _Range& r,
                           const typename basic_vec<_Idx, _IdxAbi>::mask_type& mask,
                           const basic_vec<_Idx, _IdxAbi>& indexes, flags<_Flags...> flags)
{
  using _Tp = std::ranges::range_value_t<_Range>;
  const auto rmax = r.size();

  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds("gather", indexes, rmax, mask);

  constexpr int numIndexes = basic_vec<_Idx, _IdxAbi>::size();

  // Put the indexes and the result into memory, as this generates better code
  // than trying to pull out individual elements of a simd register to pass to a
  // separate load instruction, and to put the results back into a register.
  const auto indexPtr = reinterpret_cast<const _Idx*>(&indexes);
  _Tp result[numIndexes];

  // Need simple mask which can be queried by index.
  const auto mb = mask.to_bitset();

  for (int i=0; i<numIndexes; ++i)
  {
    auto idx = indexPtr[i];
    bool validIndex = contains_flag<flag_unchecked>(flags) || (idx >= 0 && std::cmp_less(+idx, rmax));
    result[i] = validIndex && mb[i] ? r[idx] : _Tp();
  }

  return vec<_Tp, numIndexes>(result);
}

/// Scatter the contents of a simd to the supplied index positions.
/// @tparam _Range The type of the range
/// @tparam _Tp The element type
/// @tparam _TpAbi The ABI of the vec
/// @tparam _Idx The index type
/// @tparam _IdxAbi The ABI of the index simd::vec
/// @tparam _Flags The type of the flags
/// @param values The simd of values to scatter
/// @param r The range to which to scatter
/// @param indexes The simd of indexes
/// @param flags The flags controlling the scatter
template<typename _Range, typename _Tp, typename _TpAbi, typename _Idx, typename _IdxAbi, typename... _Flags>
constexpr void
scatter_to(generic_tag, const basic_vec<_Tp, _TpAbi>& values, _Range&& r,
           const basic_vec<_Idx, _IdxAbi>& indexes, flags<_Flags...> flags)
{
  static_assert(basic_vec<_Tp, _TpAbi>::size() == basic_vec<_Idx, _IdxAbi>::size(),
                "Caller should ensure sizes match");
  using _Out = std::ranges::range_value_t<_Range>;

  const auto rmax = _Idx(std::min(r.size(), size_t(std::numeric_limits<_Idx>::max())));
  const auto mb = (indexes < rmax).to_bitset();
  auto data_ptr = std::to_address(r.begin());

  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds("scatter", indexes, rmax);

  [=]<std::size_t... _Iota>(std::index_sequence<_Iota...>) {

    if constexpr (detail::contains_flag<detail::flag_unchecked>(flags))
      ((data_ptr[indexes[_Iota]] = values[_Iota]), ...);
    else
      ((mb[_Iota] ? (data_ptr[indexes[_Iota]] = values[_Iota]) : _Out()), ...);

  }(std::make_index_sequence<basic_vec<_Idx, _IdxAbi>::size>());

}

/// Scatter the contents of a simd to the supplied index positions, using a mask.
/// @tparam _Range The type of the range
/// @tparam _Tp The element type
/// @tparam _TpAbi The ABI of the vec
/// @tparam _Idx The index type
/// @tparam _IdxAbi The ABI of the index simd::vec
/// @tparam _Flags The type of the flags
/// @param values The simd of values to scatter
/// @param r The range to which to scatter
/// @param mask The mask to use for scattering
/// @param indexes The simd of indexes
/// @param flags The flags controlling the scatter
template<typename _Range, typename _Tp, typename _TpAbi, typename _Idx, typename _IdxAbi, typename... _Flags>
constexpr void
scatter_to(generic_tag, const basic_vec<_Tp, _TpAbi>& values, _Range&& r,
           const typename basic_vec<_Idx, _IdxAbi>::mask_type& mask,
           const basic_vec<_Idx, _IdxAbi>& indexes, flags<_Flags...> flags)
{
  using _Out = std::ranges::range_value_t<_Range>;
  const auto rmax = _Idx(std::min(r.size(), size_t(std::numeric_limits<_Idx>::max())));
  auto rangeCheckedMask =
    contains_flag<detail::flag_unchecked>(flags) ? mask : (indexes < rmax) && mask;
  auto data_ptr = std::to_address(r.begin());

  if constexpr (contains_flag<flag_unchecked>(flags))
    checkStaticMemoryBounds("scatter", indexes, rmax, mask);

  const auto mb = rangeCheckedMask.to_bitset();

  [=]<std::size_t... _Iota>(std::index_sequence<_Iota...>) {
    ((mb[_Iota] ? (data_ptr[indexes[_Iota]] = values[_Iota]) : _Out()), ...);
  }(std::make_index_sequence<basic_vec<_Idx, _IdxAbi>::size>());
}

/// :TODO: FMA - Don't provide generic - should always match a real instruction to get the right precision?
/// @brief Fused multiply-add for simd values.
/// @tparam _Tp The element type
/// @tparam _Abi The ABI of the vec
/// @param x The first operand
/// @param y The second operand
/// @param acc The accumulator
/// @return The result of x * y + acc
template<typename _Tp, typename _Abi>
requires (detail::is_floating_point_v<_Tp>)
constexpr basic_vec<_Tp, _Abi>
fma(generic_tag, const basic_vec<_Tp, _Abi>& x, const basic_vec<_Tp, _Abi>& y, const basic_vec<_Tp, _Abi>& acc)
{
  #if __has_builtin(__builtin_elementwise_fma)
    return __builtin_elementwise_fma(x.to_builtin(), y.to_builtin(), acc.to_builtin());
  #else
    return x.to_builtin() * y.to_builtin() + acc.to_builtin();
  #endif
}

/// @brief Fused multiply-add-subtract for simd values.
/// For all even-indexed elements compute (x * y - acc) and for all odd-indexed
/// elements compute (x * y + acc).
/// @tparam _Tp The element type
/// @tparam _Abi The ABI of the vec
/// @param x The first operand
/// @param y The second operand
/// @param acc The accumulator
/// @return The result of (x * y - acc) for even indices and (x * y + acc) for odd indices
template<typename _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi>
fmaddsub(generic_tag, const basic_vec<_Tp, _Abi>& x, const basic_vec<_Tp, _Abi>& y, const basic_vec<_Tp, _Abi>& acc)
{
  // Not the most efficient way to do this but it is meant to be overriden by
  // target-specific improvements.
  constexpr auto plusMinus = basic_vec<_Tp, _Abi>([](auto idx) -> _Tp { return (idx % 2 == 0) ? -1 : +1; });
  return fma(x, y, plusMinus * acc);
}

/// @brief Fused multiply-subtract-add for simd values.
/// For all even-indexed elements compute (x * y + acc) and for all odd-indexed
/// elements compute (x * y - acc).
/// @tparam _Tp The element type
/// @tparam _Abi The ABI of the vec
/// @param x The first operand
/// @param y The second operand
/// @param acc The accumulator
/// @return The result of (x * y + acc) for even indices and (x * y - acc) for odd indices
template<typename _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi>
fmsubadd(generic_tag, const basic_vec<_Tp, _Abi>& x, const basic_vec<_Tp, _Abi>& y, const basic_vec<_Tp, _Abi>& acc)
{
  // Not the most efficient way to do this but it is meant to be overriden by
  // target-specific improvements.
  constexpr auto minusPlus = basic_vec<_Tp, _Abi>([](auto idx) -> _Tp { return (idx % 2 == 0) ? +1 : -1; });
  return fma(x, y, minusPlus * acc);
}

/// Complex FMA. This can never be IEEE precise within 0.5ULP, but do the best possible with real FMAs to
/// get as close as possible.
/// @brief Fused multiply-add for complex simd values.
/// @tparam _Tp The floating-point type
/// @tparam _Abi The ABI of the vec
/// @param x The first operand
/// @param y The second operand
/// @param acc The accumulator
/// @return The result of x * y + acc
template<typename _Tp, typename _Abi>
inline auto fma(generic_tag,
                const basic_vec<std::complex<_Tp>, _Abi>& x,
                const basic_vec<std::complex<_Tp>, _Abi>& y,
                const basic_vec<std::complex<_Tp>, _Abi>& acc)
{
  // Convert incoming arguments into raw elements.
  const auto ex = simd_bit_cast<_Tp>(x);
  const auto ey = simd_bit_cast<_Tp>(y);

  // Generate real/imag duplicates and swaps
  const auto dupXReal = permute(ex, [](simd_size_type i) -> simd_size_type { return (i / 2) * 2; });
  const auto dupXImag = permute(ex, [](simd_size_type i) -> simd_size_type { return i | 1; });
  const auto swapY    = permute(ey, [](simd_size_type i) -> simd_size_type { return i ^ 1; });

  // Do the actual maths.
  const auto tmp = fmaddsub(dupXImag, swapY, simd_bit_cast<_Tp>(acc));
  return simd_bit_cast<std::complex<_Tp>>(fmaddsub(dupXReal, simd_bit_cast<_Tp>(y), tmp));
}

/// Conjugation for complex values.
/// @brief Conjugation for complex values.
/// @tparam _Tp The floating-point type
/// @tparam _Abi The ABI of the vec
/// @param v The simd of complex values
/// @return The conjugated simd of complex values
template<typename _Tp, typename _Abi>
constexpr auto conj(generic_tag, const basic_vec<std::complex<_Tp>, _Abi>& v) {
  // Synthesise a conjugation by setting the MSB of the complex value's container, and use that
  // to flip the imaginary sign bit.
  static_assert(detail::is_floating_point_v<_Tp>,
                "Uses floating-point trick for conjugation - need to specialise for integral later if needed");

  // XOR the floating-point value with the sign bit to flip the sign bit of the imaginaries.
  using _C = container_for_type<_Tp>;
  const auto msbImag = vec<_C, _Abi::num_elements * 2>([](auto idx) { return idx % 2 == 0 ? _C() : std::rotr(_C(1), 1); });
  const auto t = simd_bit_cast<_C>(v) ^ msbImag;

  return simd_bit_cast<std::complex<_Tp>>(t);
}

///@{
/// @brief Rotation (circular shift) operations where bits are shifted off one
/// end of each element and inserted at the other end.
/// @tparam _Tp The element type
/// @tparam _Sp The shift amount type
/// @tparam _Abi The ABI of the vec
/// @param value The value to rotate
/// @param s The offset by which to rotate
/// @return The rotated simd value
/// @internal
template<typename _Tp, std::integral _Sp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> rotate_left(generic_tag, const basic_vec<_Tp, _Abi>& value, const basic_vec<_Sp, _Abi>& s) {
  constexpr int numBits = sizeof(_Tp) * 8;
  const auto sm = basic_vec<_Tp, _Abi>(s).to_builtin() % numBits;
  const auto r = (value.to_builtin() << sm) | (value.to_builtin() >> (numBits - sm));
  return basic_vec<_Tp, _Abi>(sm != 0 ? r : value.to_builtin());
}

template<typename _Tp, std::integral _Sp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> rotate_right(generic_tag, const basic_vec<_Tp, _Abi>& value, const basic_vec<_Sp, _Abi>& s) {
  constexpr int numBits = sizeof(_Tp) * 8;
  const auto sm = basic_vec<_Tp, _Abi>(s).to_builtin() % numBits;
  const auto r = (value.to_builtin() >> sm) | (value.to_builtin() << (numBits - sm));
  return basic_vec<_Tp, _Abi>(sm != 0 ? r : value.to_builtin());
}
///@}

///@{
/// @brief Funnel shift operations where two operands are concatenated into a
/// double-width value which is then shifted, returning an N-bit window of the
/// result. funnel_shift_right extracts the low N bits; funnel_shift_left
/// extracts the high N bits.
/// @tparam _Vp The type of vec to operate on
/// @tparam _Sp The shift amount type
/// @param high The value providing the upper bits of the concatenation.
/// @param low The value providing the lower bits of the concatenation.
/// @param s The offset by which to shift. Precondition: 0 <= s[i] < N for all i.
/// @return The funnel shifted simd value
/// @internal
template<vec_unsigned_integer _Vp, vec_integral _Sp>
  requires (sizeof(typename _Vp::value_type) == sizeof(typename _Sp::value_type))
constexpr _Vp fsr(generic_tag, const _Vp& high, const _Vp& low, const _Sp& s) {
  constexpr uint8_t numBits = sizeof(typename _Vp::value_type) * 8;
  const auto sm = _Vp(s) % numBits;
  const _Vp r = (low >> sm) | (high << (numBits - sm));
  return select(sm != _Vp(), r, low);
}

template<vec_unsigned_integer _Vp, vec_integral _Sp>
  requires (sizeof(typename _Vp::value_type) == sizeof(typename _Sp::value_type))
constexpr _Vp fsl(generic_tag, const _Vp& high, const _Vp& low, const _Sp& s) {
  constexpr uint8_t numBits = sizeof(typename _Vp::value_type) * 8;
  const auto sm = _Vp(s) % numBits;
  const _Vp r = (high << sm) | (low >> (numBits - sm));
  return select(sm != _Vp(), r, high);
}
///@}

///@{
/// @brief Compute the absolute value of each element.
/// @tparam _Tp The element type
/// @tparam _Abi The ABI of the vec
/// @param v The simd value
/// @return The absolute value simd
/// Compute the absolute of a value. Typically the compiler (for at least clang
/// and oneapi) is able to take this generic implementation and turn it into a
/// native abs instruction where that exists.
template<typename _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> abs(generic_tag, const basic_vec<_Tp, _Abi>& v)
{
  if constexpr (detail::is_floating_point_v<_Tp>)
  {
    // Mask off the sign bit for floating-point. This is done explicitly so that FP16 emulation gets a fast
    // implementation. Otherwise, the conditional version below generate awful code during FP16 emulation.
    using _C = detail::container_for_type<_Tp>;
    constexpr _C msb = _C(~_C()) >> 1;
    const auto c = simd_bit_cast<_C>(v);
    auto r = basic_vec<_C, _Abi>(c & msb);
    return simd_bit_cast<_Tp>(r);
  }
  else
  {
    const auto vd = v.to_builtin();
    return basic_vec<_Tp, _Abi>(vd < _Tp() ? -vd : vd);
  }
}
///@}

/// @brief Compute the minimum or maximum of each element pair.
/// @tparam _Vp The type of vec to operate on
/// @param lhs The left-hand value
/// @param rhs The right-hand value
/// @return The elementwise minimum or maximum vec
///@{
template<vec_type _Vp>
constexpr _Vp minimum(generic_tag, const _Vp& lhs, const _Vp& rhs)
{
  #if __has_builtin(__builtin_elementwise_min)
    return __builtin_elementwise_min(lhs.to_builtin(), rhs.to_builtin());
  #else
    return select(lhs < rhs, lhs, rhs);
  #endif
}

template<vec_type _Vp>
constexpr _Vp maximum(generic_tag, const _Vp& lhs, const _Vp& rhs)
{
  #if __has_builtin(__builtin_elementwise_max)
    return __builtin_elementwise_max(lhs.to_builtin(), rhs.to_builtin());
  #else
    return select(lhs < rhs, rhs, lhs);
  #endif
}
///@}

/// @brief Saturated cast to a new type. Every element is converted to the new type, or
/// if the value can't be represented as that type then the largest or smallest value of that
/// type is used instead, whichever is closer to the original value.
/// @tparam _Up The destination type, which is constrained to being an integral value.
/// @tparam _Tp The integral type of each incoming vec element.
/// @tparam _Abi The ABI of the vec
/// @param v_in The vec value to convert.
/// @return A vec value in which each element is the saturating cast of the respective input value.
template<std::integral _To, vec_integral _From>
constexpr auto
saturating_cast(generic_tag, const _From& input) noexcept
{
  using _FromT = typename _From::value_type;
  auto value = input;

  // Clamp to the destination's minimum only when that bound is representable by
  // the source element type.Example: int8_t -> uint8_t. The destination
  // minimum, 0, fits in int8_t, so negative source values are clamped to 0.
  // Conversely, for uint8_t -> int16_t, -32768 does not fit in uint8_t and is
  // below every possible source value, so no lower clamp is needed.
  if constexpr (std::in_range<_FromT>(std::numeric_limits<_To>::min()))
    value = max(value, _From(static_cast<_FromT>(std::numeric_limits<_To>::min())));

  // Clamp to the destination's maximum only when that bound is representable by
  // the source element type. For example: uint8_t -> int8_t. The destination
  // maximum, 127, fits in uint8_t, so source values greater than 127 are
  // clamped to 127. Conversely, for int8_t -> uint8_t, 255 does not fit in
  // int8_t and is above every possible source value, so no upper clamp is
  // needed.
  if constexpr (std::in_range<_FromT>(std::numeric_limits<_To>::max()))
    value = min(value, _From(static_cast<_FromT>(std::numeric_limits<_To>::max())));

  // After the necessary clamps, every element is representable in _To, making
  // the element-wise conversion non-saturating and safe.
  return rebind_cast<_To>(value);
}

template<typename _Vendor, typename _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> byteswap(_Vendor, const basic_vec<_Tp, _Abi>& x) noexcept {

  // Break the incoming value down into register-sized pieces and reverse the
  // bytes in each piece. This is necessary because the alternative - to reverse
  // the bytes across the entire value of `x' in one permute - creates a
  // byte-element vec which could overflow gcc's limits (no more than 256
  // elements).
  auto impl = [](auto piece)
  {
    const auto asBytes = simd_bit_cast<uint8_t>(piece);
    const auto rev = permute(asBytes, [](simd_size_type idx) -> simd_size_type { return idx ^ (sizeof(_Tp) - 1); });
    return simd_bit_cast<_Tp>(rev);
  };

  return chunked_invoke(impl, x);
}

/// @brief Reverse the order of the bits in each element
/// @tparam _Vp The type of vec to operate on (must be an integral type)
/// @param x The value whose bits should be reversed.
/// @return The original value with the bits in each element reversed.
template<vec_integral _Vp> // :TODO: Should be unsigned_integral once bitswap is removed.
constexpr _Vp bitreverse(generic_tag, const _Vp& x) noexcept {

  auto doBrev = []<vec_type X>(X bits) {
    auto asBytes = simd_bit_cast<uint8_t>(bits);
    using BM = decltype(asBytes);

    constexpr BM m1(cw<0x55u>);
    constexpr BM m2(cw<0x33u>);
    constexpr BM m3(cw<0x0Fu>);

    asBytes = ((asBytes & m1) << 1u) | ((asBytes >> 1u) & m1);
    asBytes = ((asBytes & m2) << 2u) | ((asBytes >> 2u) & m2);
    asBytes = ((asBytes & m3) << 4u) | ((asBytes >> 4u) & m3);

    return byteswap(simd_bit_cast<X>(asBytes));
  };

  // Break it into small pieces to avoid creating simd objects which have too
  // many elements (e.g., a vec with 128 64-bit elements would get mapped to a
  // byte-wise permute with 1024 elements, which gcc doesn't like).
  return chunked_invoke(doBrev, x);
}

/// @brief Convert a mask into a vec value where every element is either the
/// supplied constant or zero depending upon whether the respective mask bit is
/// set or cleared.
/// @tparam _Value The value to insert wherever a mask bit is set (must be +/- 1)
/// @tparam _Tp The element type of the constructed vec
/// @tparam _Bytes The number of bytes in each element being masked
/// @tparam _Abi The ABI of the vec
/// @param mask The mask itself
/// @return A vec value in which each element is either the supplied constant or zero
///@{
template<int _Value, typename _Tp, std::size_t _Bytes, typename _Abi>
constexpr basic_vec<_Tp, _Abi> mask_to_vec(generic_tag, const basic_mask<_Bytes, _Abi>& mask)
{
  static_assert(_Value == 1 || _Value == -1, "Only some values are permitted for mask to vec output");
  static_assert(_Bytes == sizeof(_Tp), "Invalid mask or destination - differing sizes");

  if constexpr (std::integral<_Tp> && _Value == -1)
    // The mask is already entirely 0 or 1 bits, so nothing more to do
    return basic_vec<_Tp, _Abi>(mask.to_builtin());
  else if constexpr (std::integral<_Tp> && _Value == 1)
    // The mask is entirely 0 or 1 bits so a logical shift will convert the MSB into an LSB.
    return basic_vec<_Tp, _Abi>(mask.to_builtin() >> (sizeof(_Tp) * 8 - 1));
  else
  {
    constexpr auto _vs = basic_vec<_Tp, _Abi>(_Tp(_Value));
    return select(mask, _vs, basic_vec<_Tp, _Abi>());
  }
}

template<int _Value, typename _Tp, std::size_t _Bytes, typename _Abi>
constexpr basic_vec<_Tp, _Abi> mask_to_vec(compact_mask_tag, const basic_mask<_Bytes, _Abi>& m)
{
  static_assert(_Value == 1 || _Value == -1, "Only some values are permitted for mask to vec output");
  constexpr auto _vs = basic_vec<_Tp, _Abi>(_Tp(_Value));
  return select(m, _vs, basic_vec<_Tp, _Abi>());
}
///@}

/// @brief Query whether each element has exactly one bit set.
/// @tparam _Tp The element type
/// @tparam _Abi The ABI of the vec
/// @param x The vec value
/// @return A simd::mask which has an element set to true if the element has exactly one set bit.
/// @internal
template<typename _Tp, typename _Abi>
constexpr auto has_single_bit(generic_tag, const basic_vec<_Tp, _Abi>& x) noexcept {
  return (x != uint8_t(0)) && ((x & (x - uint8_t(1))) == uint8_t(0));
}

/// @brief Perform a reduction to boolean on a mask value. The functions below are correct but very likely
/// to be inefficient. It is expected that target override these with specific functions which exploit the ISA.
/// @param mask The mask to reduce
/// @return The result of the reduction (all, any, or none)
///@{
constexpr bool mask_all_of(generic_tag, const mask_type auto& mask) noexcept {
  bool allSet = true; // Assume until otherwise
  auto processBlock = [&]<mask_type _S>(_S m) {
    constexpr uint64_t allBits = (uint64_t(1) << _S::size) - 1;
    allSet &= (allBits == m.to_ullong());
  };

  chunked_invoke(processBlock, mask);
  return allSet;
}

template<mask_type _Mp>
constexpr bool mask_all_of(compact_mask_tag, const _Mp& mask) noexcept {
  return mask.to_builtin() == _Mp(true).to_builtin();
}

constexpr bool mask_any_of(generic_tag, const mask_type auto& mask) noexcept {
  // Note that real reduction (divide-and-conquor) would be faster, but only for
  // really big masks. Note also that it would be more efficient to OR the
  // values together as vecs and call to_ullong at the end, but the
  // compiler (clang/oneAPI) figures this out for itself anyway.
  uint64_t reducedBits = 0;
  chunked_invoke([&](auto m) { reducedBits |= m.to_ullong(); }, mask);
  return reducedBits != 0;
}

template<std::size_t _Bytes, typename _Abi>
constexpr bool mask_any_of(compact_mask_tag, const basic_mask<_Bytes, _Abi>& mask) noexcept {
  return mask.to_builtin() != 0; // Compact mask can be directly compared to 0.
}

constexpr bool mask_none_of(generic_tag, const mask_type auto& mask) noexcept {
  // Note that real reduction (divide-and-conquor) would be faster, but only for
  // really big masks. Note also that it would be more efficient to OR the
  // values together as vecs and call to_ullong at the end, but the
  // compiler (clang/oneAPI) figures this out for itself anyway.
  uint64_t reducedBits = 0;
  chunked_invoke([&](auto m) { reducedBits |= m.to_ullong(); }, mask);
  return reducedBits == 0;
}

template<std::size_t _Bytes, typename _Abi>
constexpr bool mask_none_of(compact_mask_tag, const basic_mask<_Bytes, _Abi>& mask) noexcept {
  return mask.to_builtin() == 0; // Compact mask can be directly compared to 0.
}
///@}

/// @brief Count the number of bits set in the mask.
/// @param mask The mask to count
/// @return The number of bits set
///@{
constexpr simd_size_type mask_reduce_count(generic_tag, const mask_type auto& mask) noexcept {
  // Note that real reduction (divide-and-conquor) would be faster, but only for
  // really big masks.
  simd_size_type numBits = 0;
  chunked_invoke([&](auto m) { numBits += std::popcount(m.to_ullong()); }, mask);
  return numBits;
}

constexpr simd_size_type mask_reduce_count(compact_mask_tag, const mask_type auto& mask) noexcept {
  // The bitset route generates code that is just as fast as anything that could be hand-written.
  return mask.to_bitset().count();
}
///@}

/// @brief Return the index of the lowest set bit in the mask.
/// @param mask The mask to query. At least one bit must be set.
/// @return The index of the lowest set bit in the mask.
///@{
constexpr simd_size_type mask_reduce_min_index(generic_tag, mask_type auto& mask) noexcept {
  simd_size_type lowestSetBit = mask.size();

  // Break the mask into individual register-sized pieces and compute the lowest
  // in each. Then merge all the results together.
  auto processBlock = [&](mask_type auto m, auto idx)
  {
    auto bits = m.to_ullong();
    if (bits != 0)
      lowestSetBit = std::min<int>(std::countr_zero(bits) + idx, lowestSetBit);
  };

  chunked_invoke(processBlock, mask);
  return lowestSetBit;
}

template<mask_type _Mp>
constexpr simd_size_type mask_reduce_min_index(compact_mask_tag, const _Mp& mask) noexcept {
  // Handling arbitrary length masks is possible, but for now the library has a
  // limit of 128 bits anyway, so exploit that to make the code simpler.
  static_assert(_Mp::size <= 128, "This function only handles up to 128 bits");

  auto lowResult = std::countr_zero((uint64_t)mask.to_builtin());

  if constexpr (_Mp::size <= 64)
    return lowResult;
  else
  {
    auto highResult = std::countr_zero((uint64_t)(mask.to_builtin() >> 64));
    if (lowResult < 64) return lowResult;
    else return highResult + 64;
  }
}
///@}

/// @brief Return the index of the highest set bit in the mask.
/// @param mask The mask to query. At least one bit must be set.
/// @return The index of the highest set bit in the mask.
///@{
constexpr simd_size_type mask_reduce_max_index(generic_tag, const mask_type auto& mask) noexcept {
  simd_size_type highestSetBit = 0;

  // Break the mask into individual register-sized pieces and compute the highest
  // in each. Then merge all the results together.
  auto processBlock = [&](mask_type auto m, auto idx) {
    auto bits = m.to_ullong();
    if (bits != 0)
      highestSetBit = std::max<int>((63 - std::countl_zero(bits)) + idx, highestSetBit);
  };

  chunked_invoke(processBlock, mask);
  return highestSetBit;
}

template<mask_type _Mp>
constexpr simd_size_type mask_reduce_max_index(compact_mask_tag, const _Mp& mask) noexcept {
  // Handling arbitrary length masks is possible, but for now the library has a
  // limit of 128 bits anyway, so exploit that to make the code simpler.
  static_assert(_Mp::size <= 128, "This function only handles up to 128 bits");

  auto lowResult = 63 - std::countl_zero((uint64_t)mask.to_builtin());

  if constexpr (_Mp::size <= 64)
    return lowResult;
  else
  {
    auto upperBits = (uint64_t)(mask.to_builtin() >> 64);
    if (upperBits != 0)
      return 127 - std::countl_zero(upperBits);
    else
      return lowResult;
  }
}
///@}

/// Create a mask which has all bits in the range [0..n) set. If n is zero then
/// no bits are set, and if N is equal or greater to the length of the mask then
/// all bits are set.
/// @tparam _Bytes the number of bytes in each element of the corresponding vec.
/// @tparam _Abi The ABI of the mask.
/// @param _n The number of required set bits in the output mask.
/// @return A mask with bits [0..n) set.
///@{
template<mask_type _Mp>
constexpr auto get_n_bit_mask(generic_tag, const _Mp&, simd_size_type _n) {
  static_assert(_Mp::size <= 256, "The code below won't work for more than 256 elements");
  if constexpr (simd_mask_element_size_v<_Mp> == 1 && _Mp::size >= 128)
  {
    // Special case. The mask's vec_type is normally signed, which for a 1 byte
    // element type is only good for up to 127 values. Handling a 128 element
    // mask means switching to unsigned bytes. Even then it can only go to 256
    // elements, but for now this will do.
    auto unsignedIota = iota<typename _Mp::traits::unsigned_vec_for_mask>;
    return unsignedIota < uint8_t(std::min<std::size_t>(_n, 255));
  } else {
    using _X = typename _Mp::traits::signed_vec_for_mask;
    return iota<_X> < (typename _X::value_type)std::min(_n, _Mp::size());
  }
}

template<mask_type _M>
constexpr auto get_n_bit_mask(compact_mask_tag, const _M&, simd_size_type _n)
{
  static_assert(_M::size <= 255);

  if constexpr (_M::size <= 64)
  {
    const auto clamped = std::min(_M::size(), _n);
    return _M::from_builtin(clamped == 64 ? ~0ULL : (1ULL << clamped) - 1);
  }
  else
    return _M::from_builtin(_M(true).to_builtin() >> (_M::size - std::min(_n, _M::size())));
}
///@}

/// @brief Count the number of leading zero bits in each element
/// @param x The input value
/// @return A vec value in which each element contains the number of leading zeros in each element.
/// @internal
template<vec_unsigned_integral _Vp>
constexpr _Vp clz(generic_tag, const _Vp& x) noexcept
  { return _Vp([=](auto i) -> typename _Vp::value_type { return std::countl_zero(x[i]); }); }

/// @brief Count the number of bits in each element.
/// @param x The input value
/// @return A vec value in which each element contains the number of 1 bits in each respective input element.
/// @internal
template<vec_unsigned_integral _Vp>
constexpr _Vp popcount(generic_tag, const _Vp& x) noexcept
  { return _Vp([=](auto i) -> typename _Vp::value_type { return std::popcount(x[i]); }); }

/// @brief Compute the saturated addition of each respective pair of vec elements
/// @ingroup simd_numeric
/// @tparam _Tp The integral type of each basic_vec element.
/// @tparam _Abi The ABI of the basic_vec
/// @param lhs The first basic_vec value
/// @param rhs The second basic_vec value
/// @return A basic_vec value in which each element is the saturating addition of the respective input basic_vec elements.
template<std::integral _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> saturating_add(generic_tag, const basic_vec<_Tp, _Abi>& lhs, const basic_vec<_Tp, _Abi>& rhs) noexcept
{
  #if __has_builtin(__builtin_elementwise_add_sat)
    return __builtin_elementwise_add_sat(lhs.to_builtin(), rhs.to_builtin());
  #else
    // Manually synthesise if the compiler can't do it. this might not generate
    // such good code as the compiler can do itself.
    if constexpr (std::unsigned_integral<_Tp>)
    {
      auto t = lhs + rhs;
      return select(t < lhs, _Tp(~_Tp()), t); // The condition is true when overflow happens (i.e., when it should saturate).
    }
    else
    {
      // Adapted from https://web.archive.org/web/20190213215419/https://locklessinc.com/articles/sat_arithmetic/
      using asUnsigned = std::make_unsigned_t<_Tp>;
      auto ux = simd_bit_cast<asUnsigned>(lhs);
      auto uy = simd_bit_cast<asUnsigned>(rhs);
	    auto res = ux + uy;

      // Create an overflow value. the top bit is shifted to give 0/1, and then
      // add the max to get min/max.
      ux = (ux >> (sizeof(_Tp) * 8 - 1)) + asUnsigned(std::numeric_limits<_Tp>::max());

      auto t = simd_bit_cast<_Tp>((ux ^ uy) | ~(ux ^ res));
      return simd_bit_cast<_Tp>(select(t >= _Tp(), ux, res));
    }
  #endif
}

/// @brief Compute the saturated subtraction of each respective pair of vec elements
/// @ingroup simd_numeric
/// @tparam _Tp The integral type of each basic_vec element.
/// @tparam _Abi The ABI of the basic_vec
/// @param lhs The first basic_vec value
/// @param rhs The second basic_vec value
/// @return A basic_vec value in which each element is the saturating subtraction of the respective input basic_vec elements.
template<std::integral _Tp, typename _Abi>
constexpr basic_vec<_Tp, _Abi> saturating_sub(generic_tag, const basic_vec<_Tp, _Abi>& lhs, const basic_vec<_Tp, _Abi>& rhs) noexcept
{
  #if __has_builtin(__builtin_elementwise_sub_sat)
    return __builtin_elementwise_sub_sat(lhs.to_builtin(), rhs.to_builtin());
  #else
    // Manually synthesise if the compiler can't do it. this might not generate
    // such good code as the compiler can do itself.
    if constexpr (std::unsigned_integral<_Tp>)
    {
      auto diff = lhs - rhs;
      return select(diff > lhs, basic_vec<_Tp, _Abi>{}, diff);
    }
    else
    {
      // Adapted from https://web.archive.org/web/20190213215419/https://locklessinc.com/articles/sat_arithmetic/
      using asUnsigned = std::make_unsigned_t<_Tp>;
      auto ux = simd_bit_cast<asUnsigned>(lhs);
      auto uy = simd_bit_cast<asUnsigned>(rhs);
      auto res = ux - uy;

      // Create an overflow value. The top bit is shifted to give 0/1, and then
      // add the max to get min/max.
      ux = (ux >> (sizeof(_Tp) * 8 - 1)) + asUnsigned(std::numeric_limits<_Tp>::max());

      auto t = simd_bit_cast<_Tp>((ux ^ uy) & (ux ^ res));
      return simd_bit_cast<_Tp>(select(t < _Tp(), ux, res));
    }
  #endif
}

/// Change the size of the elements in a mask.
/// @tparam _Result The resulting mask type
/// @tparam _Mp The input mask type
/// @param v The input mask
/// @return The converted mask
///@{
template<mask_type _To, mask_type _From>
constexpr _To convert_mask(generic_tag, const _From& v)
{
  constexpr auto fromSize = simd_mask_element_size_v<_From>;
  constexpr auto toSize = simd_mask_element_size_v<_To>;

  auto c = vec_as_container(v);

  using _ToType = container_for_num_bytes<toSize>;

  if constexpr (fromSize == toSize)
    return _To(v); // Same size, so the mask doesn't change, only its type.
  else if constexpr (toSize < fromSize)
    // Mask elements get smaller, so all ones will be truncated to still be all ones.
    return _To::from_builtin(rebind_cast<_ToType>(c).to_builtin());
  else
    // Mask elements get bigger, so do a != comparison to 0 to recover a full element.
   return rebind_cast<_ToType>(c) != _ToType();
}

template<mask_type _To, mask_type _From>
constexpr _To convert_mask(compact_mask_tag, const _From& m) { return _To::from_builtin(m.to_builtin()); }
///@}

/// Generate a mask which is suitable for storing _Np values of type _Tp. _Tp is
/// required since an individual bool needs to be turned into something suitable
/// for containing multiple bits (e.g., a float requires 32 individual bits - the
/// entire mask element is the same size as the type being represented).
///@{
template<typename _Result>
constexpr auto generate_mask_from_unsigned(generic_tag, std::unsigned_integral auto mi) {

  auto processBlock = [&]<mask_type _Mask>(_Mask, simd_size_type idx) -> _Mask {
    using _Mp = container_for_num_bytes<simd_mask_element_size_v<_Mask>>;
    using _Vp = vec<_Mp, _Mask::size>;

    auto bits = mi >> idx;

    // This code works by putting a copy of the appropriate bit into each
    // element position, and then testing that the bit is set to form a mask.
    if constexpr (sizeof(_Mp) != 1 || _Result::size <= 8)
    {
      // The simplest case can do this by broadcasting the basic bit to every
      // element. For example:
      //    abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh
      //    00000001 00000010 00000100 00001000 00010000 00100000 01000000 10000000
      // Comparing the result of the AND against zero will then set all the bits in each element.
      // This works for all sizes except bytes.
      return ((_Mp(1) << iota<_Vp>) & _Mp(bits)) != _Vp();
    }
    else
    {
      // Bytes don't work when there are more than 8 elements because you run
      // out of bits in each element to play the trick above. Instead, use the
      // following slightly more tricky code instead. This works by broadcasting
      // the lower 8 bits to the first iota lane, and the upper 8-bits to the
      // second iota lane, and so on. So 0xabcdef would become
      // [ab ab ab ab ab ab ab ab cd cd cd cd cd cd cd cd cd ef ef ef ef ef ef ef ef ef]
      const auto bytesAs64BitVec = dependent_type<_Result, vec<std::uint64_t, 1>>(bits);
      auto t = simd_bit_cast<std::uint8_t>(bytesAs64BitVec);
      auto expandedBytes = permute<_Mask::size>(t, [](int idx) { return idx / 8; });
      return ((_Mp(1) << (iota<_Vp> % _Mp(8))) & expandedBytes) != _Vp();
    }
  };

  // Create the lower elements from the incoming bits and then the upper bits
  // are filled with zeros using a resize.
  constexpr int numBits = std::min<int>(sizeof(mi) * 8, _Result::size());
  auto r = chunked_invoke(processBlock, resize_t<numBits, _Result>());
  return fit_to_size<_Result::size()>(r);
}

template<mask_type _Result>
constexpr _Result generate_mask_from_unsigned(compact_mask_tag, std::unsigned_integral auto mi) {
  // Two stage approach required to call builtin constructor. Can't just call the same
  // integral constructor for _Result since that is what called this (and which leads to recursion).
  return _Result::from_builtin(typename _Result::builtin_type(mi));
}
///@}

template<vec_floating_point _Vp>
constexpr _Vp rcp(generic_tag, const _Vp& x)
  { return uint8_t(1) / x; }

template<vec_floating_point _Vp>
constexpr _Vp sqrt(generic_tag, const _Vp& x) {
  using _Tp = typename _Vp::value_type;
  if constexpr (is_fp16_v<_Tp>)
    return _Vp([=](auto i) { return typename _Vp::value_type(std::sqrt(float(x[i]))); });
  else
    return _Vp([=](auto i) { return typename _Vp::value_type(std::sqrt(x[i])); });
}

template<vec_floating_point _Vp>
constexpr _Vp rsqrt(generic_tag, const _Vp& x)
  { return uint8_t(1) / sqrt(x); }

enum class RoundOp{CEIL, FLOOR, TRUNC, ROUND, RINT};

template<RoundOp _Op, vec_floating_point _Vp>
constexpr _Vp round_op(generic_tag, const _Vp& v)
{
  auto doOp = [](auto f) { 
    switch (_Op) {
      case RoundOp::CEIL: return std::ceil(f);
      case RoundOp::FLOOR: return std::floor(f);
      case RoundOp::TRUNC: return std::trunc(f);
      case RoundOp::ROUND: return std::round(f);
      case RoundOp::RINT: return std::rint(f);
    }
  };

  if constexpr (is_fp16_v<typename _Vp::value_type>)
    return _Vp([=](auto i) { return typename _Vp::value_type(doOp(float(v[i]))); });
  else
    return _Vp([=](auto i) { return typename _Vp::value_type(doOp(v[i])); });
}

} // namespace detail

} // namespace _XVEC_NAMESPACE::simd
