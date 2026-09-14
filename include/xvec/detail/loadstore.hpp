//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <iterator> // std::contiguous_iterator, std::iter_value_t

namespace _XVEC_NAMESPACE::simd
{

namespace detail
{
  template<typename _Vp, std::ranges::range _Range>
  using load_return_type_range = std::conditional_t<std::same_as<_Vp, std::nullptr_t>,
                                                    basic_vec<std::ranges::range_value_t<_Range>>, _Vp>;

  template<typename _Vp, std::contiguous_iterator _It>
  using load_return_type_iter = std::conditional_t<std::same_as<_Vp, std::nullptr_t>,
                                                   basic_vec<std::iter_value_t<_It>>, _Vp>;

  template<typename _Vp, std::ranges::range _Range>
  using load_mask_type_range = typename load_return_type_range<_Vp, _Range>::mask_type;

  template<typename _Vp, std::contiguous_iterator _It>
  using load_mask_type_iter = typename load_return_type_iter<_Vp, _It>::mask_type;
}

/// @brief Load the basic_vec elements from a contiguous sized range, performing
/// a suitable conversion if necessary. Two variants are provided (unmasked and
/// masked) and these provide the entry points for all the calls to the underlying
/// target-specific load implementations. Note that loading from extents with
/// known sizes are converted into special cases that handle those more
/// efficiently than a general partial load (e.g., by removing unnecessary
/// checks).
/// @ingroup simd_range
/// @tparam _Vp The vec vector to return from the load. This specifies the
/// number and type of elements. If not provided it will be deduced from the
/// range.
/// @tparam _Flags A parameter pack of types identifying properties to apply to this operation.
/// @tparam _Range The contiguous range from which to load the basic_vec values. If
/// the range has a different value type to the basic_vec then the elements will be
/// converted.
/// @param r The range from which to copy.
/// @param m A mask controlling which elements will be loaded. Elements without
/// a corresponding active mask will be default initialised.
/// @param f The flags to apply to the copy.
///@{
template<typename _Vp = std::nullptr_t, detail::contiguous_sized_range _Range, typename... _Flags>
constexpr detail::load_return_type_range<_Vp, _Range> partial_load(_Range&& r, flags<_Flags...> f = {})
{
  using _R = detail::load_return_type_range<_Vp, _Range>;
  if (std::is_constant_evaluated())
    return detail::load<_R>(generic_tag{}, std::span(r), f);
  else
  {
    constexpr auto sourceSize = detail::span_extent_v<_Range>;
    if constexpr (sourceSize == std::dynamic_extent)
      // Nothing known about the size of the source extent, so hand off to the
      // target to dynamically load what is required.
      return detail::load<_R>(target, std::span(r), f);
    else if constexpr (sourceSize == 0)
      // The source is known to be empty.
      return _R();
    else if constexpr (sourceSize <= _Vp::size())
    {
      // The source is smaller, so load what is available and default initialise
      // the rest. No checks are required because it ends up exactly the correct
      // size.
      return grow<_R::size()>(resize_t<sourceSize, _R>(r));
    }
    else
      // The source is bigger, so it is safe to load the entire vec without further checks.
      return detail::load<_R>(target, std::span(r), detail::add_unchecked_flag(f));
  }    
}

template<typename _Vp = std::nullptr_t, detail::contiguous_sized_range _Range, typename... _Flags>
constexpr detail::load_return_type_range<_Vp, _Range>
partial_load(_Range&& r, const detail::load_mask_type_range<_Vp, _Range>& m, flags<_Flags...> f = {})
{
  using _R = detail::load_return_type_range<_Vp, _Range>;
  if (std::is_constant_evaluated())
    return detail::load_masked<_R>(generic_tag{}, std::span(r), m, f);
  else
  {
    constexpr auto sourceSize = detail::span_extent_v<_Range>;
    if constexpr (sourceSize == std::dynamic_extent)
      // Nothing known about the source extents, so use the target's own version.
      return detail::load_masked<_R>(target, std::span(r), m, f);
    else if constexpr (sourceSize == 0)
      return _R();
    else if (sourceSize <= _Vp::size())
    {
      // The source is smaller, so use the mask to restrict the load size.
      // Because the load has been restricted already, the range checks can be
      // removed.
      constexpr auto shorterMask = mask_from_count<_R>(sourceSize);
      return detail::load_masked<_R>(target, std::span(r), m & shorterMask, detail::add_unchecked_flag(f));
    }
    else
      // The source is bigger, so it is safe to load the entire vec without further checks.
      return detail::load_masked<_R>(target, std::span(r), m, detail::add_unchecked_flag(f));
  } 
}
///@}

/// @brief Provide a set of overloads for different types of load operation,
/// including masked/unmasked, partial/unchecked, iterator+iterator and
/// iterator+size variants. All of these call into the two basic entry points
/// defined above.
/// @ingroup simd_range
/// @tparam _Vp The vec vector to return from the load. This specifies the
/// number and type of elements. If not provided it will be deduced from the
/// range.
/// @tparam _Flags A parameter pack of types identifying properties to apply to this operation.
/// @tparam _Range The contiguous range from which to load the basic_vec values. If
/// the range has a different value type to the basic_vec then the elements will be
/// converted.
/// @param r The range from which to copy.
/// @param m A mask controlling which elements will be loaded. Elements without
/// a corresponding active mask will be default initialised.
/// @param f The flags to apply to the copy.
///@{
template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It> partial_load(_It first, std::iter_difference_t<_It> n, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span(first, n), f); }

template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It>
partial_load(_It first, std::iter_difference_t<_It> n,
             const detail::load_mask_type_iter<_Vp, _It>& m, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span(first, n), m, f); }

template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It> partial_load(_It first, _Sent last, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span(first, static_cast<size_t>(last - first)), f); }

template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It>
partial_load(_It first, _Sent last, const detail::load_mask_type_iter<_Vp, _It>& m, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span(first, static_cast<size_t>(last - first)), m, f); }

template<typename _Vp = std::nullptr_t, detail::contiguous_sized_range _Range, typename... _Flags>
constexpr detail::load_return_type_range<_Vp, _Range> unchecked_load(_Range&& r, flags<_Flags...> f = {})
  { return partial_load<_Vp>(r, detail::add_unchecked_flag(f)); }

template<typename _Vp = std::nullptr_t, detail::contiguous_sized_range _Range, typename... _Flags>
constexpr detail::load_return_type_range<_Vp, _Range>
unchecked_load(_Range&& r, const detail::load_mask_type_range<_Vp, _Range>& m, flags<_Flags...> f = {})
  { return partial_load<_Vp>(r, m, detail::add_unchecked_flag(f)); }

template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It> unchecked_load(_It first, std::iter_difference_t<_It> n, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span(first, n), detail::add_unchecked_flag(f)); }

template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It>
unchecked_load(_It first, std::iter_difference_t<_It> n,
               const detail::load_mask_type_iter<_Vp, _It>& m, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span(first, n), m, detail::add_unchecked_flag(f)); }

template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It> unchecked_load(_It first, _Sent last, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span{first, static_cast<size_t>(last - first)}, detail::add_unchecked_flag(f)); }

template<typename _Vp = std::nullptr_t, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
constexpr detail::load_return_type_iter<_Vp, _It>
unchecked_load(_It first, _Sent last, const detail::load_mask_type_iter<_Vp, _It>& m, flags<_Flags...> f = {})
  { return partial_load<_Vp>(std::span(first, static_cast<size_t>(last - first)), m, detail::add_unchecked_flag(f)); }
///@}

/// @brief Store the elements from a basic_vec value into a contiguous range,
/// performing a suitable conversion if necessary. Both unmasked and masked
/// variants are provided. Range checking is provided to ensure that the
/// destination is not exceeded. These functions serve as the entry point for
/// all overloaded variants of vec store, including masked/unmasked,
/// iterator+iterator, iterator+count and all unchecked equivalents.
/// @tparam ..._Flags
/// @tparam _Tp The type of the element in the original source.
/// @tparam _Abi the Abi of the original vec source.
/// @tparam _Range The output range into which to store the elements. If the range has a different
/// element type then each value will be converted.
/// @param value The simd::vec value to store into the range.
/// @param r the range into which to store the simd::vec values.
/// @param flags Control any conversions.
///@{
template<typename _Tp, typename _Abi, detail::contiguous_sized_range _Range, typename... _Flags>
requires std::indirectly_writable<std::ranges::iterator_t<_Range>, _Tp>
constexpr void partial_store(const basic_vec<_Tp, _Abi>& value, _Range&& r, flags<_Flags...> f = {})
{
  if (std::is_constant_evaluated())
    detail::store(generic_tag(), value, std::span(r), f);
  else
  {
    constexpr auto destSize = detail::span_extent_v<_Range>;
    if constexpr (destSize == std::dynamic_extent)
      // Nothing known about the destination extents, so use the target's own store to handle it at runtime.
      detail::store(target, value, std::span(r), f);
    else if constexpr (destSize == 0)
      {} // Nothing to write to.
    else if constexpr  (destSize <= basic_vec<_Tp, _Abi>::size())
      // The destination is smaller so restrict the values to only those that fit.
      detail::store(target, take<destSize>(value), std::span(r), detail::add_unchecked_flag(f));
    else
      // The destination is bigger so it is safe to store the entire vec without further checks.
      detail::store(target, value, std::span(r), detail::add_unchecked_flag(f));
  }
}

template<typename _Tp, typename _Abi, detail::contiguous_sized_range _Range, typename... _Flags>
requires std::indirectly_writable<std::ranges::iterator_t<_Range>, _Tp>
constexpr void partial_store(const basic_vec<_Tp, _Abi>& value, _Range&& r,
                             const typename basic_vec<_Tp, _Abi>::mask_type& m, flags<_Flags...> f = {})
{
  if (std::is_constant_evaluated())
    detail::store_masked(generic_tag{}, value, std::span(r), m, f);
  else
  {
    constexpr auto destSize = detail::span_extent_v<_Range>;
    if constexpr (destSize == std::dynamic_extent)
      // Unknown destination size - hand off to the target to deal with the runtime behaviuor.
      detail::store_masked(target, value, std::span(r), m, f);
    else if constexpr (destSize == 0)
      {} // Nothing to write to.
    else if constexpr  (destSize <= basic_vec<_Tp, _Abi>::size())
      // The destination is smaller so restrict the values to only those that fit.
      detail::store_masked(target, take<destSize>(value), std::span(r), take<destSize>(m), detail::add_unchecked_flag(f));
    else
      // The destination is bigger so it is safe to store the entire vec without further checks.
      detail::store_masked(target, value, std::span(r), m, detail::add_unchecked_flag(f));
  }
    // 
}

///@}

/// @brief Overloads for store the elements from a basic_vec value into a variety of overloaded ranges, with and without masking.
/// @tparam ..._Flags
/// @tparam _Tp The type of the element in the original source.
/// @tparam _Abi the Abi of the original vec source.
/// @tparam _Range The output range into which to store the elements. If the range has a different
/// element type then each value will be converted.
/// @param value The simd::vec value to store into the range.
/// @param r the range into which to store the simd::vec values.
/// @param flags Control any conversions.
///@{
template<typename _Tp, typename _Abi, std::contiguous_iterator _It, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void partial_store(const basic_vec<_Tp, _Abi>& value, _It first, std::iter_difference_t<_It> n, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, n), f); }

template<typename _Tp, typename _Abi, std::contiguous_iterator _It, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void partial_store(const basic_vec<_Tp, _Abi>& value, _It first, std::iter_difference_t<_It> n,
                             const typename basic_vec<_Tp, _Abi>::mask_type& m, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, n), m, f); }

template<typename _Tp, typename _Abi, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void partial_store(const basic_vec<_Tp, _Abi>& value, _It first, _Sent last, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, static_cast<size_t>(last - first)), f); }

template<typename _Tp, typename _Abi, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void partial_store(const basic_vec<_Tp, _Abi>& value, _It first, _Sent last,
                             const typename basic_vec<_Tp, _Abi>::mask_type& m, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, static_cast<size_t>(last - first)), m, f); }

template<typename _Tp, typename _Abi, detail::contiguous_sized_range _Range, typename... _Flags>
requires std::indirectly_writable<std::ranges::iterator_t<_Range>, _Tp>
constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& value, _Range&& r, flags<_Flags...> f = {})
  { partial_store(value, std::span(r), detail::add_unchecked_flag(f)); }

template<typename _Tp, typename _Abi, detail::contiguous_sized_range _Range, typename... _Flags>
requires std::indirectly_writable<std::ranges::iterator_t<_Range>, _Tp>
constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& value, _Range&& r,
                               const typename basic_vec<_Tp, _Abi>::mask_type& m, flags<_Flags...> f = {})
  { partial_store(value, std::span(r), m, detail::add_unchecked_flag(f)); }

template<typename _Tp, typename _Abi, std::contiguous_iterator _It, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& value, _It first, std::iter_difference_t<_It> n, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, n), detail::add_unchecked_flag(f)); }

template<typename _Tp, typename _Abi, std::contiguous_iterator _It, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& value, _It first, std::iter_difference_t<_It> n,
                     const typename basic_vec<_Tp, _Abi>::mask_type& m, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, n), m, detail::add_unchecked_flag(f)); }

template<typename _Tp, typename _Abi, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& value, _It first, _Sent last, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, static_cast<size_t>(last - first)), detail::add_unchecked_flag(f)); }

template<typename _Tp, typename _Abi, std::contiguous_iterator _It, std::sized_sentinel_for<_It> _Sent, typename... _Flags>
requires std::indirectly_writable<_It, _Tp>
constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& value, _It first, _Sent last,
                               const typename basic_vec<_Tp, _Abi>::mask_type& m, flags<_Flags...> f = {})
  { partial_store(value, std::span(first, static_cast<size_t>(last - first)), m, detail::add_unchecked_flag(f)); }
///@}

} // namespace _XVEC_NAMESPACE::simd
