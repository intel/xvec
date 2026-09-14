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

template <vec_or_mask_type _Simd>
class simd_iterator
{
public:

  /// The simd object over which iteration is happening.
  _Simd* data = nullptr;

  /// The current position of the iterator within the simd being iterated.
  simd_size_type index = 0;

  // Traits
  using value_type = typename _Simd::value_type;
  using iterator_category = std::input_iterator_tag;
  using iterator_concept  = std::random_access_iterator_tag;
  using difference_type = simd_size_type;

  constexpr simd_iterator() = default;
  constexpr simd_iterator(const simd_iterator&) = default;
  constexpr simd_iterator(_Simd& s, simd_size_type x) : data(std::addressof(s)), index(x) {}

  constexpr simd_iterator& operator=(const simd_iterator&) = default;

  constexpr simd_iterator(const simd_iterator<std::remove_const_t<_Simd>>& i)
    requires std::is_const_v<_Simd> : data(i.data), index(i.index) {}

  // Dereference the iterator to get the simd element.
  constexpr value_type operator*() const { return (*data)[index]; }
  constexpr value_type operator[](difference_type i) const { return (*data)[index + i]; }

  constexpr simd_iterator& operator++() { index += 1; return *this; }
  constexpr simd_iterator& operator--() { index -= 1; return *this; }

  constexpr simd_iterator operator++(int)
  {
    simd_iterator i = *this;
    index += 1;
    return i;
  }

  constexpr simd_iterator
  operator--(int)
  {
    simd_iterator i = *this;
    index -= 1;
    return i;
  }

  constexpr difference_type operator-(simd_iterator rhs) const { return index - rhs.index; }

  friend constexpr simd_iterator   operator+(difference_type x, const simd_iterator& it) { return simd_iterator(*it.data, it.index + x); }
  friend constexpr simd_iterator   operator+(const simd_iterator& it, difference_type x) { return simd_iterator(*it.data, it.index + x); }
  friend constexpr simd_iterator   operator-(const simd_iterator& it, difference_type x) { return simd_iterator(*it.data, it.index - x); }

  constexpr simd_iterator& operator+=(difference_type x) { index += x; return *this; }
  constexpr simd_iterator& operator-=(difference_type x) { index -= x; return *this; }

  friend constexpr auto operator<=>(simd_iterator a, simd_iterator b) { return a.index <=> b.index; }
  friend constexpr bool operator==(simd_iterator a, simd_iterator b) = default;
  friend constexpr bool operator==(simd_iterator a, std::default_sentinel_t) noexcept
    { return a.index == static_cast<difference_type>(_Simd::size()); }

  friend constexpr difference_type operator-(simd_iterator a, std::default_sentinel_t) noexcept
    { return a.index - static_cast<difference_type>(_Simd::size()); }
  friend constexpr difference_type operator-(std::default_sentinel_t, simd_iterator a) noexcept
    { return static_cast<difference_type>(_Simd::size()) - a.index; }
};

}
