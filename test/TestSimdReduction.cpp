//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

/// Apply the given standard reduction to the data.
auto do_reduce(const xvec::simd::vec_or_mask_type auto& s, auto OP)
{
  auto r = s[0];
  for (int i=1; i<s.size(); ++i)
    r = OP(r, s[i]);
  return r;
}

/// Apply the given standard reduction to the masked data.
auto do_reduce(const xvec::simd::vec_or_mask_type auto& s, const xvec::simd::vec_or_mask_type auto& mask, auto identity, auto OP)
{
  decltype(identity) r = {};

  auto mask_bitset = mask.to_bitset();
  bool foundValue = false;
  for (int i=0; i<s.size(); ++i)
  {
    if (mask_bitset[i])
    {
      if (!foundValue)
        // First value is compared against identity.
        r = OP(s[i], identity);
      else
        // Once a previous value has been found, start doing the reduction.
        r = OP(r, s[i]);

      foundValue = true;
    }
  }

  return foundValue ? r : identity;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceNoMask, TypeParam, IntegerSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  BOOST_TEST(reduce(f.v0) == do_reduce(f.v0, std::plus<>{}));
  BOOST_TEST(reduce(f.v0, std::multiplies<>{}) == do_reduce(f.v0, std::multiplies<>{}));
  BOOST_TEST(reduce(f.v0, std::bit_and<>{}) == do_reduce(f.v0, std::bit_and<>{}));
  BOOST_TEST(reduce(f.v0, std::bit_or<>{}) == do_reduce(f.v0, std::bit_or<>{}));
  BOOST_TEST(reduce(f.v0, std::bit_xor<>{}) == do_reduce(f.v0, std::bit_xor<>{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceScalarNoMask, TypeParam, IntegerTypes)
{
  const TypeParam value = TypeParam(5);
  BOOST_TEST(xvec::simd::reduce(value) == value);
  BOOST_TEST(xvec::simd::reduce(value, std::multiplies<>{}) == value);
  BOOST_TEST(xvec::simd::reduce(value, std::bit_and<>{}) == value);
  BOOST_TEST(xvec::simd::reduce(value, std::bit_or<>{}) == value);
  BOOST_TEST(xvec::simd::reduce(value, std::bit_xor<>{}) == value);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceWithMaskAndIdentity, TypeParam, IntegerSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  SimdMaskFixture<typename TypeParam::mask_type> m;

  using _Tp = typename TypeParam::value_type;
  typename TypeParam::mask_type empty_mask = {};
  
  BOOST_TEST(reduce(f.v0, empty_mask, std::multiplies<>{}, _Tp(1)) == _Tp(1));
  BOOST_TEST(reduce(f.v0, empty_mask, std::bit_and<>{}, _Tp(~_Tp())) == _Tp(~_Tp()));
  BOOST_TEST(reduce(f.v0, empty_mask, std::bit_or<>{}, 0) == 0);
  BOOST_TEST(reduce(f.v0, empty_mask, std::bit_xor<>{}, 0) == 0);

  BOOST_TEST(reduce(f.v0, m.mask0, std::multiplies<>{}, _Tp(1)) == do_reduce(f.v0, m.mask0, _Tp(1), std::multiplies<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::bit_and<>{}, _Tp(~_Tp())) == do_reduce(f.v0, m.mask0, _Tp(~_Tp()), std::bit_and<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::bit_or<>{}, 0) == do_reduce(f.v0, m.mask0, 0, std::bit_or<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::bit_xor<>{}, 0) == do_reduce(f.v0, m.mask0, 0, std::bit_xor<>{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceScalarWithMaskAndIdentity, TypeParam, IntegerTypes)
{
  const TypeParam value = TypeParam(5);
  const TypeParam identity = TypeParam(7);
  BOOST_TEST(xvec::simd::reduce(value, true, std::multiplies<>{}, identity) == value);
  BOOST_TEST(xvec::simd::reduce(value, false, std::multiplies<>{}, identity) == identity);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceWithMaskAndDefaultIdentity, TypeParam, IntegerSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  SimdMaskFixture<typename TypeParam::mask_type> m;

  using _Tp = typename TypeParam::value_type;
  typename TypeParam::mask_type empty_mask = {};

  BOOST_TEST(reduce(f.v0, empty_mask) == _Tp());
  BOOST_TEST(reduce(f.v0, empty_mask, std::plus<>{}) == _Tp());
  BOOST_TEST(reduce(f.v0, empty_mask, std::multiplies<>{}) == _Tp(1));
  BOOST_TEST(reduce(f.v0, empty_mask, std::bit_and<>{}) == _Tp(~_Tp()));
  BOOST_TEST(reduce(f.v0, empty_mask, std::bit_or<>{}) == _Tp());
  BOOST_TEST(reduce(f.v0, empty_mask, std::bit_xor<>{}) == _Tp());

  BOOST_TEST(reduce(f.v0, m.mask0) == do_reduce(f.v0, m.mask0, _Tp(0), std::plus<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::plus<>{}) == do_reduce(f.v0, m.mask0, _Tp(0), std::plus<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::multiplies<>{}) == do_reduce(f.v0, m.mask0, _Tp(1), std::multiplies<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::bit_and<>{}) == do_reduce(f.v0, m.mask0, _Tp(~_Tp()), std::bit_and<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::bit_or<>{}) == do_reduce(f.v0, m.mask0, _Tp(), std::bit_or<>{}));
  BOOST_TEST(reduce(f.v0, m.mask0, std::bit_xor<>{}) == do_reduce(f.v0, m.mask0, _Tp(), std::bit_xor<>{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceScalarWithMaskAndDefaultIdentity, TypeParam, IntegerTypes)
{
  const TypeParam value = TypeParam(5);
  BOOST_TEST(xvec::simd::reduce(value, true) == value);
  BOOST_TEST(xvec::simd::reduce(value, false) == TypeParam());
  BOOST_TEST(xvec::simd::reduce(value, false, std::multiplies<>{}) == TypeParam(1));
  BOOST_TEST(xvec::simd::reduce(value, false, std::bit_and<>{}) == TypeParam(~TypeParam()));
  BOOST_TEST(xvec::simd::reduce(value, false, std::bit_or<>{}) == TypeParam());
  BOOST_TEST(xvec::simd::reduce(value, false, std::bit_xor<>{}) == TypeParam());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceMinMax, TypeParam, OrderableSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  BOOST_TEST(reduce_min(f.v0) == do_reduce(f.v0, [](auto x, auto y) { return std::min(x, y); }));
  BOOST_TEST(reduce_max(f.v0) == do_reduce(f.v0, [](auto x, auto y) { return std::max(x, y); }));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceScalarMinMax, TypeParam, ArithmeticTypes)
{
  const TypeParam value = TypeParam(5);
  BOOST_TEST(xvec::simd::reduce_min(value) == value);
  BOOST_TEST(xvec::simd::reduce_max(value) == value);
  BOOST_TEST(xvec::simd::reduce_min(value, true) == value);
  BOOST_TEST(xvec::simd::reduce_max(value, true) == value);
  BOOST_TEST(xvec::simd::reduce_min(value, false) == std::numeric_limits<TypeParam>::max());
  BOOST_TEST(xvec::simd::reduce_max(value, false) == std::numeric_limits<TypeParam>::lowest());
}

// Don't test types orderable types unless their identity can be found using numeric_limits.
template<typename _T> using HasIdentity = boost::mp11::mp_bool<std::numeric_limits<typename _T::value_type>::is_specialized>;
using OrderableWithIdentitySimdTypes = boost::mp11::mp_copy_if<OrderableSimdTypes, HasIdentity>;

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceMinMaxWithMask, TypeParam, OrderableWithIdentitySimdTypes)
{
  SimdTestFixture<TypeParam> f;
  SimdMaskFixture<typename TypeParam::mask_type> m;
  typename TypeParam::mask_type empty_mask = {};

  // Note that min and max are identical to numeric_limits, except that they
  // correctly deal with _Float16 values too. _Float16 isn't supported in
  // numeric_limits at this time.
  using _Tp = typename TypeParam::value_type;
  constexpr _Tp tmin = std::numeric_limits<_Tp>::lowest();
  constexpr _Tp tmax = std::numeric_limits<_Tp>::max();

  // Empty mask.
  BOOST_TEST(reduce_min(f.v0, empty_mask) == tmax);
  BOOST_TEST(reduce_max(f.v0, empty_mask) == tmin);

  // Non-empty mask
  BOOST_TEST(reduce_min(f.v0, m.mask0) == do_reduce(f.v0, m.mask0, tmax, [](auto x, auto y) { return std::min(x, y); }));
  BOOST_TEST(reduce_max(f.v0, m.mask0) == do_reduce(f.v0, m.mask0, tmin, [](auto x, auto y) { return std::max(x, y); }));
}

auto customMax = [](auto x, auto y) { using std::max; return max(x, y); };

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceScalarCustom, TypeParam, ArithmeticTypes)
{
  const TypeParam value = TypeParam(5);
  const TypeParam identity = TypeParam(7);
  static_assert(noexcept(xvec::simd::reduce(value, customMax)));
  static_assert(noexcept(xvec::simd::reduce(value, true, customMax, identity)));
  BOOST_TEST(xvec::simd::reduce(value, customMax) == value);
  BOOST_TEST(xvec::simd::reduce(value, true, customMax, identity) == value);
  BOOST_TEST(xvec::simd::reduce(value, false, customMax, identity) == identity);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReduceCustom, TypeParam, ArithmeticSimdTypes)
{
  SimdTestFixture<TypeParam> f;
  SimdMaskFixture<typename TypeParam::mask_type> m;
  typename TypeParam::mask_type empty_mask = {};

  auto identity = std::numeric_limits<typename TypeParam::value_type>::lowest();

  // Unmasked
  BOOST_TEST(reduce(f.v0, customMax) == do_reduce(f.v0, customMax));

  // Empty mask.
  BOOST_TEST(reduce(f.v0, empty_mask, customMax, identity) == identity);

  // Non-empty mask
  BOOST_TEST(reduce(f.v0, m.mask0, customMax, identity) == do_reduce(f.v0, m.mask0, identity, customMax));
}
