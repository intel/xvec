//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <algorithm> // std::min
#include <cstddef>
#include <tuple>
#include <type_traits>

#include <utility> // std::forward

namespace _XVEC_NAMESPACE::simd
{

namespace detail
{

/// Given a tuple, convert it into an array of values.
/// @tparam _Tuple The type of the tuple to convert
/// @param tuple The tuple value to convert
/// @return An array containing the same elements as the original tuple
template<typename _Tuple> constexpr auto get_array_from_tuple(_Tuple&& tuple) {
  constexpr auto get_array = [](auto &&...x) { return std::array{std::forward<decltype(x)>(x)...}; };
  return std::apply(get_array, std::forward<_Tuple>(tuple));
}

/// Split the incoming vec_or_mask_type value into smaller values no bigger than size N (i.e., multiple blocks of size
/// N, and possibly one remainder block less than N).
/// @tparam _Begin The starting position in the input vector to extract from
/// @tparam _Np The maximum number of elements to include in each extracted block
/// @tparam _Sp The type of the input vec or mask value
/// @param v The vec or mask value to split
/// @return A tuple of smaller vec or mask values
/// @internal
template<simd_size_type _Begin, simd_size_type _Np, vec_or_mask_type _Sp>
constexpr auto splitNHelper(const _Sp& v)
{
  constexpr simd_size_type remaining = _Sp::size - _Begin;
  constexpr simd_size_type numInBlock = std::min(remaining, _Np);

  const auto headTuple = std::make_tuple(extract<_Begin, _Begin + numInBlock>(v));

  if constexpr (remaining <= _Np)
    return headTuple;
  else
  {
    const auto tailTuple = splitNHelper<_Begin + numInBlock, _Np>(v);
    return std::tuple_cat(headTuple, tailTuple);
  }
}

/// Utility function to help concatenate several masks into a single larger mask.
/// @tparam _Begin The position at which the first mask will be inserted into the output
/// @tparam _Result The type of the output container
/// @tparam _Sp The type of the head element to be inserted
/// @param container The container that will hold the concatenated elements
/// @param head The next mask to insert
/// @param tail The remaining mask values to insert.
/// @return A new copy of the output container with all the smaller masks concatenated together.
/// @internal
template<simd_size_type _Begin, typename _Result, typename _Sp>
constexpr _Result concatHelper(const _Result& container, const _Sp& head, const vec_or_mask_type auto&... tail)
{
  _Result r;
  // Insert, permute, etc. don't work fully in constexpr contexts because they rely on compiler intrinsics
  // or implementation-specific details that are unavailable during constant evaluation. Therefore, we use
  // a fallback approach here to ensure compatibility with constexpr.
  if (std::is_constant_evaluated())
    r = _Result([=](auto i) { if (i >= _Begin && i < (_Begin + _Sp::size())) return head[i - _Begin]; else return container[i]; });
  else
    r = insert<_Begin>(container, head);

  if constexpr (sizeof...(tail) == 0)
    return r;
  else
    return concatHelper<_Begin + _Sp::size>(r, tail...);
}

}

/// Chunk the vec_or_mask_type values into a tuple of smaller values, each of which has
/// no more than N elements. The returned tuple will be a set of (size / N)
/// full-sized vec or mask values, and one element with (size % N) vec or mask values. If
/// there are no remainders then an array of equal-sized simd values will be
/// returned.
/// @tparam N The size to split into
/// @param container The svec or mask object to split
/// @return A tuple of vec or mask values of no more than N elements.
template<simd_size_type _Np, vec_or_mask_type _S>
constexpr auto
chunk(const _S& container) noexcept {
  auto r = detail::splitNHelper<0, _Np>(container);

  // Exact multiplies return as a uniform array. Non-exact multiplies have to allow the last
  // element to be a different size, so return as a tuple.
  if constexpr (_S::size % _Np == 0)
    return detail::get_array_from_tuple(r);
  else
    return r;
}

/// Chunk the vec or mask type into a tuple of smaller values, each of which has
/// no more than N elements. The returned tuple will be a set of (size / N)
/// full-sized values, and one element with (size % N) values. If
/// there are no remainders then an array of equal-sized values will be
/// returned.
/// @tparam _V The target vector type that determines the chunk size
/// @tparam _Abi The ABI type of the input vec value
/// @param s The vec object to split
/// @return A tuple of vec values of no more than _V::size elements.
///@{
template<typename _V, typename _Abi>
constexpr auto
chunk(const basic_vec<typename _V::value_type, _Abi>& s) noexcept { return chunk<_V::size>(s); }

/// @tparam _M The target mask type that determines the chunk size
/// @tparam _Abi The ABI type of the input mask value
/// @param m The mask-like object to split
/// @return A tuple of mask-like values of no more than _M::size elements.
template<mask_type _M, typename _Abi>
constexpr auto chunk(const basic_mask<simd_mask_element_size_v<_M>, _Abi>& m) noexcept  { return chunk<_M::size>(m); }
///@}

/// @brief Concatenate several smaller mask values with the same element type into
/// a single large mask value.
/// The result is the same size as the sum of the incoming containers.
/// @brief Concatenate all the specified mask values into a single mask.
/// @tparam _Tp The element type of the mask
/// @tparam _Abis Multiple parameter ABIs which specify the size of each incoming mask.
/// @param masks Multiple masks to concatenate together.
/// @return A single mask which is the same size as all the incoming
/// mask values put together, and in which their respective bits are
/// concatenated together.
template<std::size_t _Bytes, typename... _Abis>
constexpr auto cat(const basic_mask<_Bytes, _Abis>&... masks) noexcept
{
  // Determine how big to make the container to store the concatenated vec or masks, and then insert each
  // piece in turn.
  constexpr auto totalNumElements = (... + basic_mask<_Bytes, _Abis>::size());
  using ResultMask = basic_mask<_Bytes, simd_fixed_size_abi<totalNumElements>>;
  return ResultMask(detail::concatHelper<0>(ResultMask(), masks...));
}

/// @brief Concatenate several smaller vec values with the same element type into
/// a single large value.
/// The result is the same size as the concatenation of the incoming containers.
/// @tparam _Tp The element type of the vec
/// @tparam _Abis Multiple parameter ABIs which specify the size of each incoming vec.
/// @param vecs Multiple vec values to concatenate together.
/// @return A single vec which is the same size as all the incoming
/// vec values put together, and in which their respective elements are
/// concatenated together.
template<typename _Tp, typename... _Abis>
constexpr auto cat(const basic_vec<_Tp, _Abis>&... vecs) noexcept
{
  // Determine how big to make the container to store the concatenated simds, and then insert each
  // piece in turn.
  constexpr simd_size_type _resultSize = (... + basic_vec<_Tp, _Abis>::size());
  return detail::concatHelper<0>(vec<_Tp, _resultSize>(), vecs...);
}

} // namespace _XVEC_NAMESPACE::simd
