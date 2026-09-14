//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <complex>
#include <random>
#include <iostream> // Must appear before <xvec/simd> to enable operator output.
#include <utility>

#include <xvec/simd>

#include <boost/test/unit_test.hpp>

// Need some easy way to iterate through tuples in some tests.
template<std::size_t N>
void for_n(auto body)
{
    [body]<std::size_t... _Idx>(std::index_sequence<_Idx...>) {
        (body(std::integral_constant<std::size_t, _Idx>{}), ...);
    }(std::make_index_sequence<N>());
}

/// Force the argument to become constexpr. Used to force constexpr versions of
/// functions to be called.
consteval auto at_compile_time(auto x) { return x; }

/// Convert a simd::vec into an array. :TODO: or vector, carray, etc.
template <xvec::simd::vec_or_mask_type V>
constexpr auto to_array(V v)
{
  std::array<typename V::value_type, V::size> out{};
  for (std::size_t i = 0; i < V::size; ++i)
    out[i] = v[i];
  return out;
}

template<typename T, typename L>
inline T getRandomValue (L limit)
{
  static std::default_random_engine generator;

  if constexpr (requires { T().imag();}) // Complex value
    return T(getRandomValue<typename T::value_type>(limit), getRandomValue<typename T::value_type>(limit));
#if defined(__FLT16_MIN__)
  else if constexpr (xvec::simd::detail::is_fp16_v<T>)

    return _Float16(getRandomValue<float>(limit));
#endif
  // For testing user-defined types, enums, std::byte, etc, we need a way of
  // generating random numbers through a customisation point.
  else if constexpr (requires { getRandomValueForType(T(), limit);})
  {
    return getRandomValueForType(T(), limit);
  }
  else if constexpr (std::unsigned_integral<T>)
  {
    std::uniform_int_distribution<T> d(0, limit);
    return d(generator);
  }
  else if constexpr (std::integral<T>)
  {
    std::uniform_int_distribution<T> d(-limit, limit);
    return d(generator);
  }
  else
  {
    std::uniform_real_distribution<T> d(T(-limit), T(limit));
    return round(d(generator)); // Note that integer is always returned since it makes float tests less sensitive.
  }
}

// Generate a vector or random numbers, in either real or complex form. The random
// numbers will always be integer values, so that the test vector can be used in any data type
// test. A numeric limit can be provided to control the magnitude of the random data. Note
// that tests which use FP16 should choose a limit which is small enough to guarantee that
// a product and sum (i.e., FMA) can never exceed the maximum representable integer value, which for
// FP16 is 2048. The limit is therefore defaulted to 32, which is (32 * 32) + (32 * 32). Some
// tests may choose to use a tighter limit (i.e., dot product, which could add many values, and so
// each input value needs to be much smaller).
template<typename SIMD, typename LIMIT>
inline SIMD
GetRandomVector (LIMIT limit)
{
  return SIMD([=](auto) { return getRandomValue<typename SIMD::value_type>(limit); });
}

/// Generate a vec of constexpr random numbers.They aren't truly
/// random, but are good enough to suffice for checking behaviour of constexpr.
template<xvec::simd::vec_type _V, int index>
constexpr _V
GetConstexprRandomVector (int limit = 32)
{
  // Some random seeds.
  constexpr uint32_t r[]{
    0xd115236c, 0x16d16743, 0x3a697614, 0xe04e08e4,
    0xa1a53275, 0xccc10f59, 0xb95fae55, 0xecf491de,
    0x33e86773, 0x0ed24a7a, 0xd7453d80, 0x842df386,
    0x12110e76, 0x6d411a8a, 0xcbd71fed, 0x481d6b30
	};

  using _Tp = typename _V::value_type;

  if constexpr (xvec::simd::vec_complex<_V>)
  {
    // Build from real-valued elements instead.
    using REALVEC = xvec::simd::vec<typename _Tp::value_type, _V::size * 2>;
    return simd_bit_cast<typename _V::value_type>(GetConstexprRandomVector<REALVEC, index>(limit));
  }
  else if constexpr (requires { getRandomValueForType(_Tp(), limit);})
  {
    // Handle user-defined type.
    using _C = xvec::simd::detail::container_for_type<_Tp>;
    auto t = GetConstexprRandomVector<xvec::simd::vec<_C, _V::size>, index>(5);
    return simd_bit_cast<_Tp>(t);
  }
  else
  {
    // Numbers which can have negatives double the limit and subtract, to spread the values over [-limit, limit].
    constexpr bool hasNegative =
        std::signed_integral<typename _V::value_type> ||
        xvec::simd::detail::is_floating_point<typename _V::value_type>::value;

    if constexpr (hasNegative)
      limit = limit * 2;

    // Hashy-like algorithm to pick a number. Relies on the position and the incoming `seed` index.
    auto genRnd = [=](auto i) -> typename _V::value_type
    {
      auto v = (r[(i + 7) & 0xF] ^ std::rotl(r[(i + index) & 0xF], i)) % int(limit);
      if (hasNegative && ((v & 1) == 1))
        v = v - int(limit);
      return v;
    };

    return _V(genRnd);
  }
}

template<xvec::simd::mask_type _M, int index>
constexpr _M
GetConstexprRandomVector ()
{
  return _M([](auto i) { return (std::rotl(0xd115236cu, i) ^ std::rotl(0x481d6b30u, index)) & 1; });
}

/// Create a random bitset of the given length.
template<int NUM_BITS>
inline std::bitset<NUM_BITS> getRandomBitset(float probability_of_one = 0.5f) {
  static std::default_random_engine generator;
  std::bernoulli_distribution d(probability_of_one);

  std::bitset<NUM_BITS> m = {};
  for (std::size_t i=0; i<NUM_BITS; ++i)
    m.set(i, d(generator));

  return m;
}

/// Build an array which matches the given vec type and is full of iota values.
/// This assumes that the values can be constructed from an integer.
template<typename _V>
constexpr auto make_iota(int start = 0) {
  // Not all types allow a full iota, so wrap if necessary.
  int wrap = std::numeric_limits<int>::max();
  if constexpr (std::is_enum_v<typename _V::value_type>)
    wrap = 8; // Hard-coded to the maximum allowed value of any of the test enums. This won't work for arbitrary enums.

  typename std::array<typename _V::value_type, _V::size> r;
  for (int i=0; i<_V::size; ++i) r[i] = typename _V::value_type((start + i) % wrap);
  return r;
}

/// Provide access to a few default values.
template <typename _V>
class SimdTestFixture {
public:
  SimdTestFixture() = default;

  using test_array_type = std::array<typename _V::value_type, _V::size()>;

  // The random limit is used to control the range of the random numbers. It is
  // set quite low for most tests so that they work in the face of multiplies
  // and accumulations without overflowing.
  static constexpr int random_limit = 32;

  static inline _V v0 = GetRandomVector<_V>(random_limit);
  static inline _V v1 = GetRandomVector<_V>(random_limit);
  static inline _V v2 = GetRandomVector<_V>(random_limit);

  auto getRandomValue() { return ::getRandomValue<typename _V::value_type>(random_limit); }

};

/// A permute test fixture creates things to permute - both vec and mask -
/// with elements which are of all sizes, and possibly compound types (e.g.,
/// complex).
template<xvec::simd::vec_or_mask_type _S>
struct PermuteTestFixture {

  using ElementType = typename _S::value_type;

  // Note: when building complex types use the limit of the component type, not
  // the whole type.
  using ScalarLimitType = std::conditional_t<xvec::simd::detail::complex_number<ElementType>,
                                             xvec::simd::detail::complex_element_type<ElementType>, ElementType>;
  static constexpr ScalarLimitType limit = std::numeric_limits<ScalarLimitType>::max();

  PermuteTestFixture() {
    if constexpr (std::same_as<typename _S::value_type, bool>) {
      v0 = _S(getRandomBitset<_S::size>());
      v1 = _S(getRandomBitset<_S::size>());
    } else {
      v0 = GetRandomVector<_S>(limit);
      v1 = GetRandomVector<_S>(limit);
    }
  }

  _S v0, v1;
};

// Common pattern of use - compare one vec value element-wise against another
#define BOOST_SIMD_EQUAL(LHS,RHS) \
  do { static_assert(std::same_as<typename decltype(to_array(LHS))::value_type, typename decltype(to_array(RHS))::value_type>, "Expect both vecs to have same value type"); \
  BOOST_TEST(to_array(LHS) == to_array(RHS), boost::test_tools::per_element()); } while(0)

/// Given three input vectors generate a new output vector by applying a given function to the set of vec elements in turn.
/// \param simd0 The first SIMD vector to which the function will be applied.
/// \param simd1 The second SIMD vector to which the function will be applied.
/// \param simd2 The third SIMD vector to which the function will be applied.
/// \param fn A function taking one corresponding element from each vec and returning a single element.
/// \return A vector containing the result of applying the functions to each element in turn.
template<typename _T0, typename _T1, typename _T2, typename _Abi, typename FN>
auto
applyTernary (const xvec::simd::basic_vec<_T0, _Abi>& v0,
              const xvec::simd::basic_vec<_T1, _Abi>& v1,
              const xvec::simd::basic_vec<_T2, _Abi>& v2, FN fn)
{
  // Obtain the type of value generated by the function, and build a suitable result vec from it.
  using RT = decltype(fn(_T0(), _T1(), _T2()));
  ::xvec::simd::basic_vec<RT, _Abi> result = {};

  RT* const resultPtr = (RT*)&result;

  for (int i = 0; i < result.size(); ++i)
    resultPtr[i] = fn(to_array(v0)[i], to_array(v1)[i], to_array(v2)[i]);

  return result;
}

/// Given two input vectors generate a new output vector by applying a given
/// scalar function to each pair of elements in turn. The two vectors can be
/// different types provided they have the same number of elements. The result
/// takes its type from the return type of the binary function. Note that this
/// may give unexpected results if the binary result is implicitly converted
/// (e.g., uint8_t | uint8_t promotes to an int result).
/// \param lhs The left-hand side of the expression.
/// \param rhs The right-hand side of the expession.
/// \param fn A function taking two floating point values corresponding to respective elements of
/// the two inputs and which returns a result which will be inserted into the output vector.
/// \return A vector containing the result of applying the functions in turn to each pair of elements.
template<typename _T0, typename _T1, typename _Abi, typename FN>
auto
applyBinary (const xvec::simd::basic_vec<_T0, _Abi>& v0,
             const xvec::simd::basic_vec<_T1, _Abi>& v1, FN fn)
{
  // Obtain the type of value generated by the function, and build a suitable result vec from it.
  using RT = decltype(fn(_T0(), _T1()));
  ::xvec::simd::basic_vec<RT, _Abi> result = {};

  RT* const resultPtr = (RT*)&result;

  for (int i = 0; i < result.size(); ++i)
    resultPtr[i] = fn (to_array(v0)[i], to_array(v1)[i]);

  return result;
}

template<xvec::simd::vec_or_mask_type A, xvec::simd::vec_or_mask_type B, typename FN>
constexpr auto applyBinaryToArray(const A& a, const B& b, FN fn)
{
  static_assert(A::size == B::size, "applyBinaryToArray requires equal sizes");
  using Out = decltype(fn(a[0], b[0]));
  std::array<Out, A::size> out{};
  for (int i = 0; i < A::size; ++i)
    out[i] = fn(a[i], b[i]);
  return out;
}

/// Like applyBinary but all arguments and return types must be the same. Many
/// simd operators behave like this, so it is worth enforcing where needed.
template<typename _Tp, typename _Abi, typename FN>
auto
applyBinaryWithSameTypes (const xvec::simd::basic_vec<_Tp, _Abi>& v0,
                          const xvec::simd::basic_vec<_Tp, _Abi>& v1,
                          FN fn)
{
  xvec::simd::basic_vec<_Tp, _Abi> result = {};

  _Tp* const resultPtr = (_Tp*)&result;

  for (int i = 0; i < result.size(); ++i)
    // Explicitly cast the result to the correct type to avoid any quiet promotions.
    resultPtr[i] = _Tp(fn (to_array(v0)[i], to_array(v1)[i]));

  return result;
}


/// Given one input vectors generate a new output vector by applying a given function to each
/// elements in turn.
/// \param v The SIMD vector to which the function will be applied.
/// \param fn A function taking one floating point values corresponding to each element of
/// the two inputs and which returns a result which will be inserted into the output vector.
/// \return A vector containing the result of applying the functions to each element in turn.
template<typename _T, typename _Abi, typename FN>
auto
applyUnary (const xvec::simd::basic_vec<_T, _Abi>& v, FN fn)
{
  // Obtain the type of value generated by the function, and build a suitable result vec from it.
  using RT = decltype(fn(_T()));
  ::xvec::simd::basic_vec<RT, _Abi> result = {};

  RT* const resultPtr = (RT*)&result;

  for (int i = 0; i < result.size(); ++i)
    resultPtr[i] = fn(to_array(v)[i]);

  return result;
}

template<xvec::simd::vec_or_mask_type V, typename FN>
constexpr auto applyUnaryToArray(const V& v, FN fn)
{
  using Out = decltype(fn(v[0]));
  std::array<Out, V::size> out{};
  for (int i = 0; i < V::size; ++i)
    out[i] = fn(v[i]);
  return out;
}

/// Like applyBinary but all arguments and return types must be the same. Many
/// simd operators behave like this, so it is worth enforcing where needed.
template<typename _Tp, typename _Abi, typename FN>
auto
applyUnaryWithSameTypes (const xvec::simd::basic_vec<_Tp, _Abi>& v, FN fn)
{
  xvec::simd::basic_vec<_Tp, _Abi> result = {};

  _Tp* const resultPtr = (_Tp*)&result;

  for (int i = 0; i < result.size(); ++i)
    resultPtr[i] = _Tp(fn(to_array(v)[i]));

  return result;
}

#if defined(__AVX512FP16__)
inline _Float16
rsqrtHalf(_Float16 v)
{
  union {
    __m512h simd;
    _Float16 h;
  };
  h = v;
  simd = _mm512_rsqrt_ph(simd);
  return h;
}

inline _Float16
rcpHalf(_Float16 v)
{
  union {
    __m512h simd;
    _Float16 h;
  };
  h = v;
  simd = _mm512_rcp_ph(simd);
  return h;
}
#endif // FP16

/// Return the middle value from the SIMD elements when listed in ascending order.
template<typename V>
static auto
getMedian(V value)
{
  const auto container = to_array(value);
  typename decltype(container)::value_type array[V::size()];
  std::copy(container.begin(), container.end(), array);

  std::sort(array, array + container.size());
  return array[container.size() / 2];
}

/// Check the various different uses of an arbitrary operator. Note that a lambda is used to ensure
/// that the operator returns the same type as the supplied arguments - in simd no type promotion occurs.
template<typename _V, typename _OP>
static void test_binary_operator(const _V& lhs, const _V& rhs, _OP op)
{
  // Binary simd operator
  BOOST_SIMD_EQUAL(op(lhs, rhs), applyBinaryWithSameTypes(lhs, rhs, [op](auto x, auto y) { return op(x, y);}));
  BOOST_SIMD_EQUAL(op(rhs, lhs), applyBinaryWithSameTypes(rhs, lhs, [op](auto x, auto y) { return op(x, y);}));

  // Scalar on both sides of the operator.
  const auto scalar = lhs[0]; // Any value will do.
  BOOST_SIMD_EQUAL(op(scalar, rhs), applyUnaryWithSameTypes(rhs, [scalar, op](auto x) { return op(scalar, x); }));
  BOOST_SIMD_EQUAL(op(lhs, scalar), applyUnaryWithSameTypes(lhs, [scalar, op](auto x) { return op(x, scalar); }));
}

template<int _Begin, int _End, size_t _Size>
std::bitset<_End - _Begin> getBitSubset(const std::bitset<_Size>& bits)
{
  constexpr int resultSize = _End - _Begin;
  std::bitset<resultSize> result;
  for (std::size_t i=0; i<resultSize; ++i)
    result[i] = bits[i + _Begin];
  return result;
}

template<int _Begin, int _End, size_t _OriginalSize, size_t _InsertedSize>
std::bitset<_OriginalSize> setBitSubset(const std::bitset<_OriginalSize>& original, const std::bitset<_InsertedSize>& new_bits)
{
  static_assert(_End - _Begin <= _InsertedSize, "Can't insert into a region which is bigger than the supplied insertion bits");
  std::bitset<_OriginalSize> result = original;
  for (std::size_t i=0; i<(_End - _Begin); ++i)
    result[i + _Begin] = new_bits[i];
  return result;
}

/// Create a bultin mask which matches the given bitset.
template<typename BUILTIN_MASK, int NUM_ELEMENTS>
static decltype(auto) getWideMask(std::bitset<NUM_ELEMENTS> bits) {
  BUILTIN_MASK m = {};

  for (std::size_t i=0; i<NUM_ELEMENTS; ++i)
    m[i] = bits[i];

  // The mask is full of zero or not zero, so now comparing it directly to zero will fill it with
  // the appropriate all 0's or all 1's.
  typename std::remove_reference<decltype(m[0])>::type zero = {};

  return BUILTIN_MASK(m != zero);
}

/// Create a random narrow (integer) mask (Intel AVX-512 style).
template<typename BUILTIN_MASK, int NUM_BITS>
static BUILTIN_MASK getNarrowMask(std::bitset<NUM_BITS> bits) {
  // Convert the bitset into a compact mask, bit-by-bit.
  BUILTIN_MASK result = {};
  for (std::size_t i=0; i<bits.size(); ++i)
    result |= (BUILTIN_MASK(bits[i]) << i);
  return result;
}

/// Return a random mask of the appropriate type.
template<typename BUILTIN_MASK, int NUM_BITS>
static BUILTIN_MASK getMaskFromBitset(std::bitset<NUM_BITS> bits) {
  if constexpr (xvec::simd::has_compact_mask<xvec::simd::target_tag>)
    return getNarrowMask<BUILTIN_MASK, NUM_BITS>(bits);
  else
    return getWideMask<BUILTIN_MASK, NUM_BITS>(bits);
}

template <typename T> struct SimdMaskFixture {
  // Rely on the bitset giving the correct number of bits to use. There is no other easy way to do
  // it, since the mask may contain fewer bits than the builtin would suggest (e.g., 64-bits in Intel SSE
  // would be 2 bits, but the smallest __mmask8 is too big and the number of bits would be wrong).
  static constexpr int k_numBits = T::size();

  // Create some random masks. Start by generating a random bitset, and then convert it to the corresponding
  // builtin, and finally into the mask itself. This way the mask is built from its components, so that various
  // tests can make sure that turning it back into the other components works.
  const std::bitset<k_numBits> bitset0 = getRandomBitset<k_numBits>();
  const typename T::builtin_type builtin0 = getMaskFromBitset<typename T::builtin_type, k_numBits>(bitset0);
  const T mask0 = T::from_builtin(builtin0);

  const std::bitset<k_numBits> bitset1 = getRandomBitset<k_numBits>();
  const typename T::builtin_type builtin1 = getMaskFromBitset<typename T::builtin_type, k_numBits>(bitset1);
  const T mask1 = T::from_builtin(builtin1);

  /// Create a constexpr mask of every 3rd bit, which can be used for constexpr tests.
  static constexpr auto every3 = T([](auto i) { return i % 3 == 0; });

};

namespace std {
  inline std::ostream &operator<<(std::ostream &stream, __int128) {
    stream << "CannotPrintInt128";
    return stream;
  }

  inline std::ostream &operator<<(std::ostream &stream, unsigned __int128) {
    stream << "CannotPrintInt128";
    return stream;
  }

#if defined(__FLT16_MIN__) && !defined(__STDCPP_FLOAT16_T__)
  inline std::ostream &operator<<(std::ostream &stream, _Float16 f) {
    stream << float(f);
    return stream;
  }
#endif

}

#if defined(__FLT16_MIN__) && !defined(__STDCPP_FLOAT16_T__)

// Provide direct access to boost to print out _Float16
namespace boost { namespace test_tools { namespace tt_detail {

template<>
struct print_log_value<_Float16> {
    void operator()(std::ostream& os, _Float16 v) {
        os << static_cast<float>(v);   // or however you want to format it
    }
};

}}} // namespace boost::test_tools::tt_detail

#endif // FP16
