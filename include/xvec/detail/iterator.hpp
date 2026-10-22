//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <compare>
#include <iterator> // std::random_access_iterator_tag
#include <type_traits>

namespace _XVEC_NAMESPACE::simd
{

/**
 * @defgroup cpp26_iterator C++26: SIMD iteration
 * @brief Random-access iteration over vector and mask elements.
 *
 * @see https://eel.is/c++draft/simd.iterator
 * @{
 */

/// @brief xvec implementation of the exposition-only SIMD iterator.
/// @tparam _Simd The vector or mask type being traversed.
template <vec_or_mask_type _Simd>
class simd_iterator
{
public:

  /// The simd object over which iteration is happening.
  /// @internal
  _Simd* data = nullptr;

  /// The current position of the iterator within the simd being iterated.
  /// @internal
  simd_size_type index = 0;

  // Traits
  using value_type = typename _Simd::value_type;
  using iterator_category = std::input_iterator_tag;
  using iterator_concept  = std::random_access_iterator_tag;
  using difference_type = simd_size_type;

  /// Default-construct an iterator.
  constexpr simd_iterator() = default;

  /// Copy an iterator.
  constexpr simd_iterator(const simd_iterator&) = default;

  /// Construct an iterator at element x of s.
  constexpr simd_iterator(_Simd& s, simd_size_type x) : data(std::addressof(s)), index(x) {}

  /// Copy-assign an iterator.
  constexpr simd_iterator& operator=(const simd_iterator&) = default;

  /// Convert a mutable iterator to a const iterator.
  constexpr simd_iterator(const simd_iterator<std::remove_const_t<_Simd>>& i)
    requires std::is_const_v<_Simd> : data(i.data), index(i.index) {}

  /// Return the current element.
  constexpr value_type operator*() const { return (*data)[index]; }
  /// Return the element at the specified relative offset.
  constexpr value_type operator[](difference_type i) const { return (*data)[index + i]; }

  /// Advance to the next element.
  constexpr simd_iterator& operator++() { index += 1; return *this; }

  /// Move to the previous element.
  constexpr simd_iterator& operator--() { index -= 1; return *this; }

  /// Advance to the next element, returning the original iterator.
  constexpr simd_iterator operator++(int)
  {
    simd_iterator i = *this;
    index += 1;
    return i;
  }

  /// Move to the previous element, returning the original iterator.
  constexpr simd_iterator
  operator--(int)
  {
    simd_iterator i = *this;
    index -= 1;
    return i;
  }

  /// Return the distance from another iterator.
  constexpr difference_type operator-(simd_iterator rhs) const { return index - rhs.index; }

  /// Return an iterator advanced by the specified offset.
  friend constexpr simd_iterator   operator+(difference_type x, const simd_iterator& it) { return simd_iterator(*it.data, it.index + x); }

  /// Return an iterator advanced by the specified offset.
  friend constexpr simd_iterator   operator+(const simd_iterator& it, difference_type x) { return simd_iterator(*it.data, it.index + x); }

  /// Return an iterator moved backward by the specified offset.
  friend constexpr simd_iterator   operator-(const simd_iterator& it, difference_type x) { return simd_iterator(*it.data, it.index - x); }

  /// Advance by the specified offset.
  constexpr simd_iterator& operator+=(difference_type x) { index += x; return *this; }

  /// Move backward by the specified offset.
  constexpr simd_iterator& operator-=(difference_type x) { index -= x; return *this; }

  /// Compare iterator positions.
  friend constexpr auto operator<=>(simd_iterator a, simd_iterator b) { return a.index <=> b.index; }

  /// Test iterator equality.
  friend constexpr bool operator==(simd_iterator a, simd_iterator b) = default;

  /// Test whether the iterator has reached the end.
  friend constexpr bool operator==(simd_iterator a, std::default_sentinel_t) noexcept
    { return a.index == static_cast<difference_type>(_Simd::size()); }

  /// Return the signed distance from the end sentinel.
  friend constexpr difference_type operator-(simd_iterator a, std::default_sentinel_t) noexcept
    { return a.index - static_cast<difference_type>(_Simd::size()); }
  /// Return the number of elements remaining before the end sentinel.
  friend constexpr difference_type operator-(std::default_sentinel_t, simd_iterator a) noexcept
    { return static_cast<difference_type>(_Simd::size()) - a.index; }
};

/// @} cpp26_iterator

}
