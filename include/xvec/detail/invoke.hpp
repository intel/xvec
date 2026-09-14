//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <utility> // std::forward, std::integer_sequence

namespace _XVEC_NAMESPACE::simd
{

namespace detail
{
/// Compute the native size of a vec or mask type.
/// _Tp::never_instantiated is a trick to make sure we never instantiate
/// primary template
template<typename _Tp>
inline constexpr auto arg_native_size = _Tp::never_instantiated;

template<typename _Tp, typename _Abi>
inline constexpr auto arg_native_size<basic_vec<_Tp, _Abi>> = detail::max_native_size<_Tp>;

template<std::size_t _Bytes, typename _Abi>
inline constexpr auto arg_native_size<basic_mask<_Bytes, _Abi>> = detail::maxBytesInVec / _Bytes;

/// Checks that all sizes are equal.
template <simd_size_type Size, simd_size_type... Sizes>
inline constexpr bool are_all_sizes_equal = (... && (Size == Sizes));

} // End namespace detail

template<simd_size_type _SuggestedN = 0, typename _Fn, vec_or_mask_type _Arg0, vec_or_mask_type... _Args>
constexpr auto chunked_invoke(_Fn fn, const _Arg0& arg0, const _Args&... args)
{
  using namespace detail;

  // Check that all sizes of vec or mask arguments are equal element-wise
  static_assert(are_all_sizes_equal<_Arg0::size(), _Args::size()...>, "chunked_invoke expects all vec or mask arguments to have the same size");
   
  // Check that block size is either set by the user or all native sizes are equal
  static_assert(_SuggestedN != 0 || are_all_sizes_equal<arg_native_size<_Arg0>, arg_native_size<_Args>...>, "chunked_invoke expects all vec or mask arguments to have the same native size. Use the first template parameter to override this");

  // Get the size of each piece which will be processed.
  constexpr auto _B = _SuggestedN == 0 ? arg_native_size<_Arg0> : _SuggestedN;

  // Check the function appears to be callable, and whether it takes an index or not.
  constexpr auto _TS = std::min<simd_size_type>(_B, _Arg0::size());
  using ArgTypesWithoutIndex = std::tuple<resize_t<_TS, _Arg0>, resize_t<_TS, _Args>...>;
  using ArgTypesWithIndex = std::tuple<resize_t<_TS, _Arg0>, resize_t<_TS, _Args>..., simd_size_type>;
  constexpr bool callableWithoutIndex = requires { fn(extract<0, _TS>(arg0), extract<0, _TS>(args)...); };
  constexpr bool callableWithIndex = requires { fn(extract<0, _TS>(arg0), extract<0, _TS>(args)..., simd_size_type(0)); };
  static_assert(callableWithIndex || callableWithoutIndex, "chunked_invoke can't call the function");
  using ArgTypes = std::conditional_t<callableWithIndex, ArgTypesWithIndex, ArgTypesWithoutIndex>;

  // Query the return type and check that it appears valid. The result must
  // either be void, or a vec or mask type that can be joined to the results of the
  // other pieces.
  using ResultType = decltype(std::apply(fn, ArgTypes()));
  constexpr bool isVoidFn = std::same_as<void, ResultType>;
  static_assert(isVoidFn || vec_or_mask_type<ResultType>, "chunked_invoke expects the function to return a vec or mask result, or none at all");

  // This utility lambda calls the function with a specific region of each vec or mask argument.
  auto callFn = [&]<simd_size_type _Begin>(size_constant<_Begin>) {
    constexpr int _End = std::min<int>(_Begin + _B, _Arg0::size());
    if constexpr (callableWithoutIndex)
      return fn(extract<_Begin, _End>(arg0), extract<_Begin, _End>(args)...);
    else
      return fn(extract<_Begin, _End>(arg0), extract<_Begin, _End>(args)..., _Begin);
  };

  // Call each sub-block as many times as needed.
  constexpr int numBlocks = (_Arg0::size() + (_B - 1)) / _B;
  return [=]<simd_size_type... _Idx>(std::integer_sequence<simd_size_type, _Idx...>)
  {
    // Note that the blocks must be executed in order of index.
    if constexpr (isVoidFn)
      (callFn(size_constant<_Idx * _B>{}), ...);
    else
    {
      // Brace-init guarantees order.
      auto results = std::tuple { callFn(size_constant<_Idx * _B>{})... };

      // Concatenate the results together
      return std::apply([](auto&&... r) { return cat(std::forward<decltype(r)>(r)...); }, results);
    }
  }(std::make_integer_sequence<simd_size_type, numBlocks>{});

}

} // namespace _XVEC_NAMESPACE::simd
