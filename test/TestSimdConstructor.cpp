//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

// Some of the checks work on types which can be converted from short int.
template<typename T> using is_convertible_from_ushort =
  mp11::mp_bool<std::convertible_to<unsigned short, typename T::value_type> && !xvec::simd::detail::complex_number<typename T::value_type>>;
using ConvertibleFromUnsignedShortSimdTypes = mp11::mp_copy_if<AllSimdTypes, is_convertible_from_ushort>;

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstExprFromBuiltin, TypeParam, AllSimdTypes)
{
  // Any constexpr values, really. But they must be the type of the elements of the underlying builtin.
  constexpr auto k = 34;
  constexpr typename TypeParam::builtin_type v = {k};
  using BaseElementType = std::remove_reference_t<decltype(v[0])>;

  constexpr TypeParam tp(v); // constexpr from a builtin type

  // Build an array from an initialiser list with the same form. We expect the same behaviour from
  // the vec initialiser (e.g., zero fill unspecified elements).
  const BaseElementType expectedArray[sizeof(v) / sizeof(BaseElementType)] = {k};
  const auto expected = xvec::simd::partial_load<TypeParam>((typename TypeParam::value_type*)expectedArray, TypeParam::size);
  BOOST_SIMD_EQUAL(tp, expected);
}

void CheckImplicitConstructor()
{
  // Check that various implicit constructions are possible. For example, small
  // to large float is allowed implicitly, but large to small float must be
  // explicit.
  using VS = xvec::simd::vec<short, 8>;
  using VI = xvec::simd::vec<int, 8>;
  using VF = xvec::simd::vec<float, 8>;
  using VD = xvec::simd::vec<double, 8>;

  // Allowed implicit conversions.
  [[maybe_unused]] VS xss = VS(); // short to short
  [[maybe_unused]] VI xsi = VS(); // short to int
  [[maybe_unused]] VF xsf = VS(); // short to float
  [[maybe_unused]] VD xsd = VS(); // short to double
  [[maybe_unused]] VD xii = VI(); // int to int
  [[maybe_unused]] VD xid = VI(); // int to double
  [[maybe_unused]] VF xff = VF(); // float to float
  [[maybe_unused]] VD xfd = VF(); // float to double

  // Disallowed implicit conversions.
  static_assert(!std::is_convertible_v<VI, VS>); // int to short
  static_assert(!std::is_convertible_v<VF, VS>); // float to short
  static_assert(!std::is_convertible_v<VD, VS>); // double to short
  static_assert(!std::is_convertible_v<VF, VI>); // float to int
  static_assert(!std::is_convertible_v<VD, VI>); // double to int
  static_assert(!std::is_convertible_v<VD, VF>); // double to float
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromScalarZero, TypeParam, AllSimdTypes)
{
  const typename TypeParam::value_type k_zero = {};
  TypeParam zero = TypeParam(k_zero);

  constexpr typename TypeParam::value_type expected[TypeParam::size()] = {};
  BOOST_TEST(to_array(zero) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InitialiseFromScalarNonZero, TypeParam, AllSimdTypes)
{
  SimdTestFixture<TypeParam> f;

  const auto value = f.getRandomValue();
  TypeParam simd = TypeParam(value);

  // Non-constexpr
  {
    typename TypeParam::value_type expected[TypeParam::size()] = {};
    std::fill(std::begin(expected), std::end(expected), value);
    BOOST_TEST(to_array(simd) == expected, boost::test_tools::per_element());
  }

  // Constexpr
  {
    constexpr auto value = typename TypeParam::value_type(9); // Arbitrary value
    typename TypeParam::value_type expected[TypeParam::size()] = {};
    std::fill(std::begin(expected), std::end(expected), value);

    constexpr auto simd = TypeParam(value);
    BOOST_TEST(to_array(simd) == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FromSameSimdType, TypeParam, ArithmeticSimdTypes)
{
  // Everything can be copied from a signed char.
  constexpr auto t = xvec::simd::iota<TypeParam>;

  const auto expected = make_iota<TypeParam>();

  // Converting copies don't work yet because llvm has issues with constexpr, but this makes sure that at least
  // copies into the same type will work, which is useful for constexpr versions of other functions.
  constexpr auto computedConstexpr = TypeParam(t);
  BOOST_TEST(to_array(computedConstexpr) == expected, boost::test_tools::per_element());

  auto computedDynamic = TypeParam(std::as_const(t));
  BOOST_TEST(to_array(computedDynamic) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FromOtherSimdType, TypeParam, ArithmeticSimdTypes)
{
  // Everything can be copied from a signed char.
  constexpr auto t = xvec::simd::iota<xvec::simd::vec<int8_t, TypeParam::size()>>;

  typename SimdTestFixture<TypeParam>::test_array_type expected = {};
  std::iota(expected.begin(), expected.end(), 0);

  // Issue - no constexpr support
  // constexpr auto computedConstexpr = TypeParam(t);
  // BOOST_TEST(to_array(computedConstexpr), ElementsAreArray(expected));

  auto computedDynamic = TypeParam(std::as_const(t));
  BOOST_TEST(to_array(computedDynamic) == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructorFromRangeOfWrongSize, TypeParam, AllSimdTypes)
{
  std::array<unsigned short, TypeParam::size + 1> input;
  static_assert(! requires { TypeParam(input); }, "Shouldn't be able to build from an array known to be the wrong size");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructorFromStaticExtent, TypeParam, ConvertibleFromUnsignedShortSimdTypes)
{
  // An array of unsigned short integer values to load from. Note that although this
  // isn't a value preservable type (i.e., unsigned short can't be represented
  // in all loadable types without loss), the values below are all valid. The
  // flag_convert parameter prevents this from being an error.
  std::array<unsigned short, TypeParam::size> input;
  std::iota(input.begin(), input.end(), 3);

  // The expected values to load
  typename SimdTestFixture<TypeParam>::test_array_type expected;
  for (size_t i=0; i<expected.size(); ++i) expected[i] = typename TypeParam::value_type(i + 3);

  TypeParam computed(input, xvec::simd::flag_convert);
  BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());

  // Check for implicit conversion. This only works on value preserving types.
  if constexpr (xvec::simd::detail::value_preserving_convertible_to<unsigned short, typename TypeParam::value_type>)
  {
    constexpr TypeParam zero = {};
    auto result = zero + input; // Second argument is an array, but it is being added to a vec.
    BOOST_TEST(to_array(result) == expected, boost::test_tools::per_element());
  }

}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructorFromMaskedStaticExtent, TypeParam, ConvertibleFromUnsignedShortSimdTypes)
{
  // An array of unsigned short integer values to load from. Note that although this
  // isn't a value preservable type (i.e., unsigned short can't be represented
  // in all loadable types without loss), the values below are all valid. The
  // flag_convert parameter prevents this from being an error.
  std::array<unsigned short, TypeParam::size> input;
  std::iota(input.begin(), input.end(), 3);

  // The expected values to load. Remove any odd elements to compensate for the mask.
  typename SimdTestFixture<TypeParam>::test_array_type expected;
  for (size_t i=0; i<expected.size(); ++i) expected[i] = typename TypeParam::value_type(i + 3);
  for (std::size_t i=0; i<TypeParam::size(); ++i)
    if (i % 2 != 0)
      expected[i] = typename TypeParam::value_type{};

  constexpr typename TypeParam::mask_type evenMask([](size_t idx) -> bool { return (idx % 2) == 0;});

  TypeParam computed(input, evenMask, xvec::simd::flag_convert);
  BOOST_TEST(to_array(computed) == expected, boost::test_tools::per_element());

  // :TODO: Constexpr version
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstructorFromStaticCtad, TypeParam, ArithmeticTypes)
{
  // Note that TypeParam is a scalar value, not vec. The size is fixed for this test.

  // C++ Array CTAD.
  std::array<TypeParam, 10> arrayInput;
  std::iota(arrayInput.begin(), arrayInput.end(), 3);
  xvec::simd::basic_vec computedFromArray = arrayInput; // CTAD - no arguments specified but it figures out size and type from the array.
  // xvec::simd::simd computed = arrayInput; Waiting for clang/OneAPI to be fixed to allow this too.
  BOOST_TEST(to_array(computedFromArray) == arrayInput, boost::test_tools::per_element());

  // Span CTAD.
  std::span spanInput = arrayInput;
  xvec::simd::basic_vec computedFromSpan = spanInput; // CTAD - no arguments specified but it figures out size and type from the span.
  // xvec::simd::simd computed = input; Waiting for clang/OneAPI to be fixed to allow this too.
  BOOST_TEST(to_array(computedFromSpan) == spanInput, boost::test_tools::per_element());

  // C-style array
  TypeParam carrayInput[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  xvec::simd::basic_vec computedFromCarray = carrayInput; // CTAD - no arguments specified but it figures out size and type from the c-array.
  // xvec::simd::simd computed = input; Waiting for clang/OneAPI to be fixed to allow this too.
  BOOST_TEST(to_array(computedFromCarray) == carrayInput, boost::test_tools::per_element());

  // With an extra parameter.
  auto computedCtad = xvec::simd::basic_vec(spanInput, xvec::simd::flag_convert);
  BOOST_TEST(to_array(computedCtad) == spanInput, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprConstructorFromStaticCtad, TypeParam, ArithmeticTypes)
{
  // Note that TypeParam is a scalar value, not vec. The size is fixed for this test.
  static constexpr auto arrayInput = std::array<TypeParam, 10>{4, 2, 3, 8, 7, 56, 4, 5, 9, 10};

  // C++ Array CTAD.
  constexpr xvec::simd::basic_vec computedFromArray = arrayInput;
  BOOST_TEST(to_array(computedFromArray) == arrayInput, boost::test_tools::per_element());

  // Span CTAD.
  constexpr auto spanInput = std::span{arrayInput};
  constexpr xvec::simd::basic_vec computedFromSpan = spanInput; // CTAD - no arguments specified but it figures out size and type from the span.
  BOOST_TEST(to_array(computedFromSpan) == arrayInput, boost::test_tools::per_element());

  // // C-style array
  constexpr TypeParam carrayInput[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  constexpr xvec::simd::basic_vec computedFromCarray = carrayInput; // CTAD - no arguments specified but it figures out size and type from the c-array.
  BOOST_TEST(to_array(computedFromCarray) == carrayInput, boost::test_tools::per_element());
}

#if defined(_XVEC_HAS_CONSTEXPR)
BOOST_AUTO_TEST_CASE_TEMPLATE(ConstexprConstructorFromMaskedStaticCtad, TypeParam, ArithmeticTypes)
{
  // Note that TypeParam is a scalar value, not vec. The size is fixed for this test.

  static constexpr auto arrayInput = std::array<TypeParam, 10>{4, 2, 3, 8, 7, 56, 4, 5, 9, 10};
  constexpr auto mask = xvec::simd::mask<TypeParam, 10>([](auto i) { return i % 2 != 0; });

  auto expected = std::array<TypeParam, 10>{0, 2, 0, 8, 0, 56, 0, 5, 0, 10}; // Mask out some elements.

  using SIMD = xvec::simd::vec<TypeParam, 10>;

  // C++ Array (no ctad for masked)
  constexpr auto computedFromArray = SIMD(arrayInput, mask);
  BOOST_TEST(to_array(computedFromArray) == expected, boost::test_tools::per_element());

  // Span (no ctad for masked)
  constexpr auto spanInput = std::span{arrayInput};
  constexpr auto computedFromSpan = SIMD(spanInput, mask);
  BOOST_TEST(to_array(computedFromSpan) == expected, boost::test_tools::per_element());

  // // C-style array (no ctad for masked)
  constexpr TypeParam carrayInput[10] = {0, 2, 0, 8, 0, 56, 0, 5, 0, 10};
  constexpr auto computedFromCarray = SIMD(carrayInput, mask);
  BOOST_TEST(to_array(computedFromCarray) == expected, boost::test_tools::per_element());
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(Generator, TypeParam, AllSimdTypes)
{
  // Generate values to fill the vec using the following function, which
  // converts an index into a value. Note the expression is chosen to use values
  // which will not overrun any of the enum test types.
  constexpr auto gen = [](size_t idx) { return typename TypeParam::value_type((idx % 7) + 2); };

  typename SimdTestFixture<TypeParam>::test_array_type expected = {};
  for (size_t i=0; i<TypeParam::size(); ++i)
    expected[i] = gen(i);

  const auto computedDynamic = TypeParam(gen);
  BOOST_TEST(to_array(computedDynamic) == expected, boost::test_tools::per_element());

  constexpr auto computedConstexpr = TypeParam(gen);
  BOOST_TEST(to_array(computedConstexpr) == expected, boost::test_tools::per_element());
}

void CheckDisallowedGenerators()
{
  struct S {
    operator double() const;       // basic_vec(U&& value)
    double operator()(int) const;  // basic_vec(G&& gen)
    double* begin() const;         // basic_vec(R&& r, flags<Flags...> = {});
    double* end() const;
    constexpr static int size() { return 2; }
  };

  static_assert(!xvec::simd::detail::generator_invocable<S, double, 8>);
}

