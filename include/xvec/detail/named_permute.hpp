//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <utility> // std::index_sequence, std::make_index_sequence

namespace _XVEC_NAMESPACE::simd {

namespace detail {

  /// @brief A named permute which keeps the first _Np values of a simd and discards the remainder.
  /// @tparam _Np The number of elements to keep.
  template<int _Np, vec_or_mask_type _Sp>
  constexpr auto permute_take(generic_tag, const _Sp& in) {
    static_assert(_Np <= _Sp::size, "Too short to take N elements");
    return permute<_Np>(in, [](simd_size_type i) -> simd_size_type { return i ; });
  }

  /// @brief A named permute which discards the first _Np values of a vec or mask, returning the remainder.
  /// @tparam _Np The number of elements to discard.
  template<int _Np, vec_or_mask_type _Sp>
  constexpr auto permute_drop(generic_tag, const _Sp& in) noexcept {
    static_assert(_Sp::size >= _Np, "Too short to drop N elements");
    return permute<_Sp::size - _Np>(in, [](simd_size_type i) -> simd_size_type { return i + _Np; });
  }

  /// @brief A named permute which creates N copies of the input concatenated together.
  /// @tparam _Np The number of repeats of the input to create.
  // :TODO: Or concat several?
  template<int _Np, vec_or_mask_type _Sp>
  constexpr auto permute_repeat_all(generic_tag, const _Sp& in) noexcept {
    static_assert(_Np >= 1, "Expected at least one repeat");
    return permute<_Sp::size * _Np>(in, [](simd_size_type i) -> simd_size_type { return i % _Sp::size; });
  }

  /// @brief A named permute which creates N copies of each element.
  /// @tparam _Np The number of repeats of each element.
  template<int _Np, vec_or_mask_type _Sp>
  constexpr auto permute_repeat_each(generic_tag, const _Sp& in) noexcept {
    static_assert(_Np >= 1, "Expected at least one repeat");
    return permute<_Sp::size * _Np>(in, [](simd_size_type i) -> simd_size_type { return i / _Np; });
  }

  /// @brief A named permute which extracts every N'th value, discarding values which do not have indexes which are multiples of N.
  /// @tparam _Np The multiple of the element indexes to read from.
  template<int _Np, vec_or_mask_type _Sp>
  constexpr auto permute_stride(generic_tag, const _Sp& in)noexcept {
    constexpr int NewSize = (_Sp::size + (_Np - 1)) / _Np;
    return permute<NewSize>(in, [](simd_size_type i) -> simd_size_type { return i * _Np ; });
  }

  /// @brief A named permute which increases the number of elements in the
  /// vec or mask, placing a copy of the given value into the new element positions.
  template<int _Np, vec_or_mask_type _Sp>
  constexpr auto permute_grow(generic_tag, const _Sp& in) noexcept {
    static_assert(_Np >= _Sp::size, "Can't grow to a size smaller than the input");
    return permute<_Np>(in, [](simd_size_type i) -> simd_size_type { return i < _Sp::size ? i : zero_element; });
  }

  // Grow with filled initialise
  template<int _Np, vec_type _Vp>
  constexpr auto permute_grow(generic_tag, const _Vp& in, typename _Vp::value_type fill) noexcept {
    static_assert(_Np >= _Vp::size, "Can't grow to a size smaller than the input");
    using _NewType = resize_t<_Np, _Vp>;
    using _M = typename _NewType::mask_type;
    return select(_M::__mask_from_count(_Vp::size), permute_grow<_Np>(target, in), _NewType(fill));
  }

  template<int _Np, mask_type _Mp>
  constexpr auto permute_grow(generic_tag, const _Mp& in, bool fill) noexcept {
    static_assert(_Np >= _Mp::size, "Can't grow to a size smaller than the input");
    using _NewMask = resize_t<_Np, _Mp>;
    return select(_NewMask::__mask_from_count(_Mp::size), permute_grow<_Np>(target, in), _NewMask(fill));
  }

  /// @brief Reverse the order in which the elements appear in the given input vec or mask.
  template<vec_or_mask_type _Sp>
  constexpr auto permute_reverse(generic_tag, const _Sp& in) {
    return permute(in, [](simd_size_type i) -> simd_size_type { return (_Sp::size - 1) - i; });
  }

  /// @brief A named permute which rotates the input so that the element at the MIDDLE index becomes the first element.
  /// @tparam _Middle The _Middle index, which will be moved to the first index.
  template<int _Middle, vec_or_mask_type _Sp>
  constexpr auto permute_rotate(generic_tag, const _Sp& in) noexcept {
    static_assert(_Middle >= 0 && _Middle < _Sp::size, "Expected middle to refer to valid index [0..size)");
    return permute(in, [](simd_size_type i) -> simd_size_type { return (i + _Middle) % _Sp::size; });
  }

  template<int ROWS, int COLS, vec_or_mask_type _Sp>
  constexpr auto permute_transpose(generic_tag, const _Sp& in) noexcept {
    static_assert(ROWS * COLS == _Sp::size, "Input data is not a matrix of size ROWS * COLS");
    return permute(in, [](auto idx) { return (idx / ROWS) + (idx % ROWS) * COLS; });
  }

  /// zip takes one or more vec or mask values and produces a new vec or mask where all the
  /// i'th elements are moved into contiguous positions. For example, zip([a b
  /// c], [0 1 2]) becomes  [a 0 b 1 c 2]. The output size is limited by the
  /// smallest of the inputs.
  template<vec_or_mask_type... _Sps>
  constexpr auto permute_zip(generic_tag, const _Sps& ...simds) noexcept {
    constexpr auto ROWS = sizeof...(simds);
    constexpr auto COLS = std::min({_Sps::size()...});
    auto t = cat(permute_take<COLS>(target, simds)...);
    return permute<ROWS * COLS>(t, [](auto idx) { return (idx / ROWS) + (idx % ROWS) * COLS; });
  }

  /// @brief Unzip breaks a vec or mask down into N smaller objects, where each vec or mask is the
  /// values at the stride of N and offset by the output index. For example,
  /// unzip<2>([a 0 b 1 c 2]) will become [a b c] [0 1 2].
  /// @tparam _Np The number of simds to create.
  template<int _Np, vec_or_mask_type _Sp>
  constexpr auto permute_unzip(generic_tag, const _Sp& in) noexcept {
    static_assert(_Sp::size >= _Np, "Must have at least one element per output");
    return [=]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
      return std::make_tuple(stride<_Np>(permute_drop<_Idx>(target, in))...);
    } (std::make_index_sequence<_Np>());
  }

  /// Return a copy of the incoming vec_or_mask_type value, with the element at the given index replaced with the new value.
  ///@{
  template<vec_type _Vp>
  constexpr _Vp set_element(generic_tag, const _Vp& v, int index, typename _Vp::value_type new_value)
  {
    using _C = detail::container_for_type<typename _Vp::value_type>;

    auto t = vec_as_container(v).to_builtin(); // Index using the raw compiler support. 
    t[index] = std::bit_cast<_C>(new_value);

    return std::bit_cast<_Vp>(rebind_t<_C, _Vp>(t)); // Reconstruct back to original type.
  }

  template<mask_type _Mp>
  constexpr _Mp
  set_element(generic_tag, const _Mp& m, int index, bool new_value) {
    auto t = m.to_builtin();
    t[index] = new_value;
    return _Mp::from_builtin(t);
  }

  template<mask_type _Mp>
  constexpr _Mp
  set_element(compact_mask_tag, const _Mp& m, int index, bool new_value) {
    auto maskBit = typename _Mp::builtin_type(1) << index;
    auto newBit = new_value << index;
    return _Mp::from_builtin((m.to_builtin() & ~maskBit) | newBit);
  }
  ///@}
  
  template<simd_size_type _Begin, simd_size_type _End, vec_or_mask_type _Vp>
  constexpr auto extract(generic_tag, const _Vp& v) {
    static_assert(_Begin < _End, "Start of requested range must be less than end");
    static_assert(_End <= _Vp::size(), "Requested range can't overlap end of container");
    return permute<_End - _Begin>(v, [](int i) -> int { return i + _Begin; });
  }

  template<simd_size_type _Begin, simd_size_type _End, mask_type _Mp>
  constexpr auto extract(compact_mask_tag, const _Mp& m)
  {
    static_assert(_Begin < _End, "Start of requested range must be less than end");
    static_assert(_End <= _Mp::size(), "Requested range can't overlap end of container");

    // Move the desired part to the lower bits, and then fit it to the desired
    // output size.
    return take<_End - _Begin>(_Mp::from_builtin(m.to_register() >> _Begin));
  }

  template<simd_size_type _Begin, vec_or_mask_type _Container, vec_or_mask_type _Child>
  requires (vec_or_mask_same_element_type<_Container, _Child>)
  constexpr _Container
  insert(generic_tag, const _Container& container, const _Child& child) noexcept
  {
    static_assert((_Child::size() + _Begin) <= _Container::size(), "Child cannot overlap container end");

    // Resize the child to be the same size as the container. Permute only works if they are the same size.
    const auto resizedChild = permute<_Container::size()>(child, detail::perm_uninitResize);

    // Create a sequence which is a copy of the parent, but with a sub-sequence referencing the child.
    auto inserter = [](int i) -> int {
      if ((i >= _Begin) && i < (_Begin + _Child::size()))
        return (i - _Begin) + _Container::size();              // Choose from the child.
      else
        return i;                                          // Choose from the parent
    };

    return permute_pair<_Container::size()>(container, resizedChild, inserter);
  }

  template<simd_size_type _Begin, mask_type _Container, mask_type _Child>
  requires (vec_or_mask_same_element_type<_Container, _Child>)
  constexpr _Container
  insert(compact_mask_tag, const _Container& container, const _Child& child) noexcept
  {
    static_assert((_Child::size() + _Begin) <= _Container::size(), "Child cannot overlap container end");

    if constexpr (_Container::size() == _Child::size())
      return child; // Some cases of cat need this. :TODO: Fix cat instead.
    else
    {
      using _C = typename _Container::builtin_type;
      // Create a block of bits that is positioned at the insertion position.
      constexpr auto block = ((_C(1) << _Child::size()) - 1) << _Begin;

      // Zero out the insertion position.
      const auto removeBlock = container.to_builtin() & ~block;

      // Move the desired bits to the insertion position and limit to just those bits.
      const auto insertBlock = (_C(child.to_builtin()) << _Begin) & block;

      return _Container::from_builtin(insertBlock | removeBlock);
    }
  }

}

/// Keep the first _Np elements of the input, discarding the remainder. This
/// is equivalent to shrinking the vec or mask to a new size.
template<int _Np> constexpr auto take(const vec_or_mask_type auto& in) { return detail::permute_take<_Np>(target, in); }

/// Discard the first _Np elements of the input, discarding the remainder.
template<int _Np> constexpr auto drop(const vec_or_mask_type auto& in) { return detail::permute_drop<_Np>(target, in); }

/// Grow the vec or mask input to be N elements.
template<int _Np> constexpr auto grow(const vec_or_mask_type auto& in) { return detail::permute_grow<_Np>(target, in); }
template<int _Np, vec_or_mask_type _Sp> constexpr auto grow(const _Sp& in, const typename _Sp::value_type& fill)
{
  using _NewType = resize_t<_Np, _Sp>;
  if (std::is_constant_evaluated())
    return _NewType([=](auto i) { return i < _Sp::size ? in[i] : fill; });
  else
    // :TODO: Make select/mask_from_count constexpr?
    return detail::permute_grow<_Np>(target, in, fill);
}

/// Repeat the entire input value N times. The input [a b c] repeated 3 times would be [a b c a b c a b c]
template<int _Np> constexpr auto repeat_all(const vec_or_mask_type auto& in) { return detail::permute_repeat_all<_Np>(target, in); }

/// Repeat each value in the vec or mask N times. The input [a b c] repeated 2 times would be [a a b b c c]
template<int _Np> constexpr auto repeat_each(const vec_or_mask_type auto& in) { return detail::permute_repeat_each<_Np>(target, in); }

/// Reverse the order of the elements in the input vec or mask.
constexpr auto reverse(const vec_or_mask_type auto& in) { return detail::permute_reverse(target, in); }

/// Extract every element which is a multiple of N, discarding the others. To
/// have an offset stride use drop first.
template<int _Np, vec_or_mask_type _V> constexpr auto stride(const _V& in) {
  if (std::is_constant_evaluated())
    return permute<(_V::size + (_Np - 1)) / _Np>(in, [](simd_size_type i) -> simd_size_type { return i * _Np ; });
  else
    return detail::permute_stride<_Np>(target, in); }

/// Rotate the elements within the vec or mask such that the element at index MIDDLE
/// is moved to the first index position. The MIDDLE index must be in the
/// range [0..SIZE).
template<int _Middle> constexpr auto rotate(const vec_or_mask_type auto& in) { return detail::permute_rotate<_Middle>(target, in); }

/// zip takes one or more vec or mask values and produces a new object where all the
/// i'th elements are moved into contiguous positions, for all values of i
/// between [0..min_size_of_inputs). For example, zip([a b c d e], [0 1 2])
/// becomes  [a 0 b 1 c 2]. Note the discard of [d e] to match the size of [0
/// 1 2].
template<vec_or_mask_type... _Sps> constexpr auto zip(const _Sps& ...simds) { return detail::permute_zip(target, simds...); }

/// unzip breaks a vec or mask down into N smaller objects, where each object is the
/// values at the stride of N and offset by the output simd index. For example,
/// unzip<2>([a 0 b 1 c 2]) will become [a b c] [0 1 2].
template<int N> constexpr auto unzip(const vec_or_mask_type auto& in) { return detail::permute_unzip<N>(target, in); }

/// Treat the vec or mask as a matrix of size [ROWS, COLS] and transpose the data into a matrix of size[COLS, ROWS].
template<int _Rows, int _Cols> constexpr auto transpose(const vec_or_mask_type auto& in) { return detail::permute_transpose<_Rows, _Cols>(target, in); }

/// Insert a scalar element into the given index position of the supplied vec_or_mask_type object.
template<vec_or_mask_type _V> constexpr auto set_element(const _V& v, int index, typename _V::value_type element)
{
  if (std::is_constant_evaluated())  
    return _V([=](auto i) { return i == index ? element : v[i]; });
  else
    return detail::set_element(target, v, index, element);
}

/// Extract the subset denoted by the range [begin..end) from the supplied vec or mask.
template<int _Begin, int _End, vec_or_mask_type _Vp>
constexpr resize_t<_End - _Begin, _Vp> extract(const _Vp& v) { return detail::extract<_Begin, _End>(target, v); }

/// Return a copy of the parent basic_vec such that the selected range of elements
/// is overwritten by the respective elements from the child basic_vec.
/// @tparam _Begin The compile-time index at which to insert the child
/// @tparam _Parent The type of the parent container.
/// @tparam _Child The type of the child container.
/// @param parent The parent container.
/// @param child The child container.
/// @return A copy of the original container with its original element values in
/// the position [_Begin, _Begin + child::size()] replaced by the values from
/// the child. @{
template<simd_size_type _Begin, vec_or_mask_type _Parent, vec_or_mask_type _Child>
requires (vec_or_mask_same_element_type<_Parent, _Child>)
constexpr _Parent insert(const _Parent& parent, const _Child& child) noexcept
  { return detail::insert<_Begin>(target, parent, child); }

} // namespace _XVEC_NAMESPACE::simd
