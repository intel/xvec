//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <cstddef>
#include <limits> // std::numeric_limits

namespace _XVEC_NAMESPACE::simd
{

namespace detail
{
inline auto minBinaryOperator = [](auto lhs, auto rhs) { using std::min; return min(lhs, rhs); };
inline auto maxBinaryOperator = [](auto lhs, auto rhs) { using std::max; return max(lhs, rhs); };

template<typename _Op, typename _Tp>
  concept binary_reduction_fn =
    requires (const _Op binary_op, const vec<_Tp, 1> v) {{ binary_op(v, v) } -> std::same_as<vec<_Tp, 1>>; };

template<typename _Op, typename _Tp>
constexpr auto choose_identity_element() {
  if constexpr (std::same_as<_Op, std::plus<>>)
    return _Tp();
  else if constexpr (std::same_as<_Op, std::multiplies<>>)
    return _Tp(1);
  else if constexpr (std::same_as<_Op, std::bit_and<>>)
    return _Tp(~_Tp());
  else if constexpr (std::same_as<_Op, std::bit_or<>>)
    return _Tp();
  else if constexpr (std::same_as<_Op, std::bit_xor<>>)
    return _Tp();
  else
    static_assert(dependent_false<_Op>, "Unsupported binary reduction operator");
}

/// Reduction helper function. This works by splitting down into two
/// approximately equal power-of-2 sized pairs, and then applying the binary
/// operator, before recursively applying. Note that the size is passed in
/// explicitly so that when the reduction has dropped below the minimum register
/// size, the incoming argument can remain a simd register.
/// \internal
template<simd_size_type _Size, typename _Tp, typename _Abi, binary_reduction_fn<_Tp> _BinaryFn>
inline _Tp reduce_helper(const basic_vec<_Tp, _Abi>& v, _BinaryFn binop) {

  if constexpr (_Size == 1)
    return v[0];
  else
  {
    // Split the complete incoming simd into two pieces: one a power-of-2 and
    // the other the remainder, where both are at least as big as the smallest
    // register. It is better to handle anything small as a complete register,
    //     rather than extracting and inserting sub-register pieces in the final
    // stages of the reduction.
    constexpr int halfSize = std::bit_ceil((uint64_t)_Size) / 2;
    constexpr auto opSize = std::max(register_size<_Tp, 1>, halfSize);

    const auto lower = permute<opSize>(v, perm_uninitResize);
    const auto upper = permute<opSize>(drop<halfSize>(v), perm_uninitResize);

    auto r = binop(lower, upper);

    // If the upper and lower are different sizes, because the incoming value
    // was not divisible by 2, then mask off the unwanted elements and copy the
    // originals through.
    constexpr auto remainderSize = _Size - halfSize;
    if constexpr (halfSize != remainderSize)
    {
      const auto remainderMask = mask<_Tp, opSize>::__mask_from_count(remainderSize);
      r = select(remainderMask, r, lower);
    }

    return reduce_helper<halfSize>(r, binop);
  }
}

} // End namespace detail

/// @brief Apply a reduction operation across all elements of the basic_vec value to reduce them to
/// a single scalar value.
/// @tparam _Tp The element type of the basic_vec to reduce
/// @tparam _Abi The ABI of the basic_vec to reduce
/// @tparam _BinaryFn The type of function to apply
/// @param v The basic_vec value to reduce into a single element
/// @param binop The binary operation to use to generate the reduction (defaults to std::plus)
/// @return A scalar value resulting from repeatedly applying the binary operation to all values in the
/// basic_vec in some arbitrary order.
template<vec_type _V, detail::binary_reduction_fn<typename _V::value_type> _BinaryFn = std::plus<>>
constexpr typename _V::value_type reduce(const _V& v, _BinaryFn binop = {})
  { return detail::reduce_helper<_V::size()>(v, binop); }

/// @brief Apply a reduction operation across all elements of the basic_vec value, using a mask and identity element.
/// @tparam _Tp The element type of the basic_vec to reduce
/// @tparam _Abi The ABI of the basic_vec to reduce
/// @tparam _BinaryFn The type of function to apply
/// @param v The basic_vec value to reduce
/// @param m The mask indicating which elements to include in the reduction
/// @param binop The binary operation to use to generate the reduction (defaults to std::plus)
/// @param identity_element The identity value for the reduction operation
/// @return A scalar value resulting from repeatedly applying the binary operation to all selected values in the basic_vec
template<typename _Tp, typename _Abi, detail::binary_reduction_fn<_Tp> _BinaryFn = std::plus<>>
constexpr _Tp reduce(const basic_vec<_Tp, _Abi>& v,
                     const typename basic_vec<_Tp, _Abi>::mask_type& m,
                     _BinaryFn binop = {},
                     std::type_identity_t<_Tp> identity_element = detail::choose_identity_element<_BinaryFn, _Tp>())
  { return detail::reduce_helper<_Abi::num_elements>(select(m, v, identity_element), binop); }

/// @brief Apply a reduction operation to a scalar value.
/// @tparam _Tp The scalar value type.
/// @tparam _BinaryFn The type of function to apply.
/// @param v The scalar value to reduce.
/// @param binop The binary operation for the reduction.
/// @return The input value.
template<detail::vectorizable _Tp,
         detail::binary_reduction_fn<_Tp> _BinaryFn = std::plus<>>
constexpr _Tp reduce(const _Tp& v, [[maybe_unused]] _BinaryFn binop = {}) noexcept
  { return v; }

/// @brief Apply a reduction operation to a conditionally selected scalar value.
/// @tparam _Tp The scalar value type.
/// @tparam _BinaryFn The type of function to apply.
/// @param v The scalar value to reduce.
/// @param m Whether the scalar value is selected.
/// @param binop The binary operation for the reduction.
/// @param identity_element The identity value for the reduction operation.
/// @return The input value when selected; otherwise, the identity element.
template<detail::vectorizable _Tp,
         detail::binary_reduction_fn<_Tp> _BinaryFn = std::plus<>>
constexpr _Tp reduce(
    const _Tp& v, std::same_as<bool> auto m, [[maybe_unused]] _BinaryFn binop = {},
    std::type_identity_t<_Tp> identity_element =
        detail::choose_identity_element<_BinaryFn, _Tp>()) noexcept
  { return m ? v : identity_element; }

/// @brief Reduction operators for specific types of pre-defined reduction
/// @{
/// @tparam _Tp The element type of the basic_vec.
/// @tparam _Abi The ABI of the basic_vec
/// @param v The basic_vec value to reduce
/// @return A single value of type _Tp representing the reduction.
template<vec_type _V> requires requires (_V x) { min(x, x); }
constexpr typename _V::value_type reduce_min(const _V& v) noexcept
  { return reduce(v, detail::minBinaryOperator); }

template<vec_type _V> requires requires (_V x) { reduce_min(x); }
constexpr  typename _V::value_type
reduce_min(const _V& v, const typename _V::mask_type& m) noexcept
{
  // Note that it is more efficient to replace unused elements with the max
  // value as that eliminates any need to do masking during the reduction.
  const auto r = select(m, v, std::numeric_limits<_V>::max());
  return reduce_min(r);
}

template<vec_type _V> requires requires (_V x) { max(x, x); }
constexpr typename _V::value_type reduce_max(const _V& v) noexcept { return reduce(v, detail::maxBinaryOperator); }

template<vec_type _V> requires requires (_V x) { reduce_max(x); }
constexpr typename _V::value_type
reduce_max(const _V& v, const typename _V::mask_type& m) noexcept
{
  // Note that it is more efficient to replace unused elements with the lowest
  // value as that eliminates any need to do masking during the reduction.
  const auto r = select(m, v, std::numeric_limits<_V>::lowest());
  return reduce_max(r);
}

/// @brief Return the minimum of a scalar value.
/// @tparam _Tp The scalar value type.
/// @param v The scalar value.
/// @return The input value.
template<detail::vectorizable _Tp> requires std::totally_ordered<_Tp>
constexpr _Tp reduce_min(const _Tp& v) noexcept { return v; }

/// @brief Return the minimum of a conditionally selected scalar value.
/// @tparam _Tp The scalar value type.
/// @param v The scalar value.
/// @param m Whether the scalar value is selected.
/// @return The input value when selected; otherwise, the maximum value of _Tp.
template<detail::vectorizable _Tp> requires std::totally_ordered<_Tp>
constexpr _Tp reduce_min(const _Tp& v, std::same_as<bool> auto m) noexcept
  { return m ? v : std::numeric_limits<_Tp>::max(); }

/// @brief Return the maximum of a scalar value.
/// @tparam _Tp The scalar value type.
/// @param v The scalar value.
/// @return The input value.
template<detail::vectorizable _Tp> requires std::totally_ordered<_Tp>
constexpr _Tp reduce_max(const _Tp& v) noexcept { return v; }

/// @brief Return the maximum of a conditionally selected scalar value.
/// @tparam _Tp The scalar value type.
/// @param v The scalar value.
/// @param m Whether the scalar value is selected.
/// @return The input value when selected; otherwise, the lowest value of _Tp.
template<detail::vectorizable _Tp> requires std::totally_ordered<_Tp>
constexpr _Tp reduce_max(const _Tp& v, std::same_as<bool> auto m) noexcept
  { return m ? v : std::numeric_limits<_Tp>::lowest(); }
///@}

} // namespace _XVEC_NAMESPACE::simd
