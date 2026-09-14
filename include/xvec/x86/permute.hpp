//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xvec/detail/config.hpp>

#include <cstdint> // std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t
#include <utility> // std::index_sequence, std::make_index_sequence

namespace _XVEC_NAMESPACE::simd {

namespace detail
{

// Utility to duplicate the bits in the input (e.g., 0101 becomes 00110011).
// this is primarily used in Intel AVX-512 for vec elements which are bigger
// than 64-bit, and hence require several blocks of 64-bit elements to be used.
// It is no great pain then to use pdef, since most of the Intel AVX-512
// instructions will have that readily available.
#if defined(__BMI2__)
inline uint64_t dupBits (uint32_t bits) {
  auto spreadBits = _pdep_u64(uint64_t(bits), 0x55555555);
  return spreadBits | (spreadBits << 1);
}
#endif

/// Special permutation code for dealing with permutions of very large elements which
/// have no native support. The elements are handled by treating them as blocks
/// of 64-bit element instead, and permuting each element using severla such
/// indexes.
template<typename _Tp, typename _AbiT, typename _Ip, typename _AbiI>
constexpr auto permute_large_elements(const basic_vec<_Tp, _AbiT>& values, const basic_vec<_Ip, _AbiI>& indexes)
{
  static_assert(sizeof(_Tp) == 16, "Assume 128-bit is the maximum to deal with for now");

  // :TODO: It is likely there are special cases that can be done faster. Or
  // rather than duplicating the indexes, we could make the element smaller
  // chunks and then recombine. Lots of room to explore. Just do enough to get
  // it working for now.

  // :TODO: Work harder to make it an amenable format to begin with? Convert to 64-bit indexes for example?

  // Convert each index into a pair of 64-bit indexes which permute the upper
  // and lower parts of the input elements. Given an index [3, 5, 1] then the
  // 64-bit indexes would be [6, 7, 10, 11, 2, 3] for example (n => [n*2,
  // n*2+1]).
  auto bigIdxs = chunked_invoke([]<typename _Vec>(_Vec s) {
    // Duplicate every index -> [3 3 5 5 1 1]
    auto dup = permute<_Vec::size * 2>(s, [](auto idx) { return idx / 2; });
    constexpr auto oddOnes = 
      vec<typename _Vec::value_type, _Vec::size * 2>([](auto i) -> typename _Vec::value_type { return i & 1; });
    return (dup * cw<2>) + oddOnes;
  }, indexes);

  // Now do the permute again using 64-bit elements with the new indexes.
  auto r = permute(simd_bit_cast<uint64_t>(values), bigIdxs);

  // Map back to the original 128-bit elements.
  return simd_bit_cast<_Tp>(r);
}


/// Permute individual bytes using the in-lane shuffle instruction. This is faster
/// than permutex[2]var_epi8 where it exists for small use cases, and on
/// machines which don't support permutex[2]var_epi8 extended variants of this which duplicate the source values are likely to be faster too.
template<typename _Tp, typename _AbiT, typename _Ip, typename _AbiI>
constexpr auto permute_byte_shuffle (const basic_vec<_Tp, _AbiT>& v, const basic_vec<_Ip, _AbiI>& indexes)
{
  static_assert(sizeof(_Tp) == 1 && sizeof(_Ip) == 1);
  static_assert(basic_vec<_Tp, _AbiT>::size <= 16 && basic_vec<_Ip, _AbiI>::size <= 16, "Data and indexes must fit first lane");
  // :TODO: In future both indexes and values could extend beyond lanes through
  // duplication or multi-lane index merging, which will be faster than using permutex[2]var_epi8.
  return vec<_Tp, basic_vec<_Ip, _AbiI>::size>(_mm_shuffle_epi8(v.to_register(), indexes.to_register() & 0xf));
}

/// Permute one register of Intel AVX2 values.
template<std::unsigned_integral _Tp, typename _AbiT, std::integral _Ip, typename _AbiI>
requires (sizeof(basic_vec<_Tp, _AbiT>) <= 32)
inline auto permute_x86_register(x86_avx2_tag, const basic_vec<_Tp, _AbiT>& v, const basic_vec<_Ip, _AbiI>& indexes)
{
  // Bump the incoming values up to a whole register so that the _mm256_permute
  // instruction can be used. Both the indexes and the values need to be the
  // same size. There is no advantage to using 128-bit permute instructions.
  const auto _rv = permute<vec<_Tp>::size()>(v, perm_uninitResize);

  auto impl = [_rv]<typename _ImplT>(_ImplT i) -> _ImplT {
    auto block = permute<vec<_Tp>::size>(i, perm_uninitResize); // Bump to whole register

    auto r = [=]{
      if constexpr (sizeof(_Tp) == 4)
        return _mm256_permutevar8x32_epi32(__m256i(_rv.to_register()), block.to_register());
      else if constexpr (sizeof(_Tp) == 8)
      {
        // There is no easy dynamic pd permute, so translate it into a 32-bit shuffle of pair-wise values instead.
        // :TODO: permutevar_pd could be used, but it is rather awkward. Probably faster though, even then.
        const auto dup32 = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(block.to_register()))); // Dup 32-bit index
        const auto mulBy2 = _mm256_slli_epi32(dup32, 1);
        const auto idx32 = _mm256_add_epi64(mulBy2, __v4di() + 0x100000000);
        return _mm256_permutevar8x32_epi32(_rv.to_register(), idx32);
      }
      else
        static_assert(dependent_false<_Tp>, "Unhandled source type for permute");
    }();

    return fit_to_size<_ImplT::size>(vec<_Tp>(r));
  };

  return chunked_invoke(impl, indexes);
}

#if defined(__AVX512BW__) && defined(__AVX512F__)
/// A 2-source 8-bit register permutation emulation used on machines which don't have native support.
inline __m512i
_mm512_permutex2var_epi8_emulated(__m512i v_data_0,
                                  __m512i v_shuf_idxs,
                                  __m512i v_data_1)
{
  // No 8-bit shuffle available, so use 16-bit instead, and process the upper and lower bytes in
  // each 16-bit block independently. Note that each index is 6-bit.

  // Each pair of 8-bit indexes is used to find the 16-bit block which will hold the value of interest in either
  // the upper or lower byte. The odd and even indexes each need their own permutation, and the index to use
  // for that permutation comes from the upper 5-bits of the incoming index (i.e., discard the LSB). The permute
  // instruction masks the index bits itself, so we only need to put the upper 5-bits of the index
  // into the correct location.
  __v32hi evenPerm = _mm512_permutex2var_epi16(v_data_0, v_shuf_idxs >> 1, v_data_1);
  __v32hi oddPerm = _mm512_permutex2var_epi16(v_data_0, v_shuf_idxs >> 9, v_data_1);

  // evenPerm and oddPerm now contain pairs of bytes. We need to select which byte of each pair to use,
  // and that selection is driven by the LSB of each index (i.e., LSB == 0 means use the lower byte of the
  // pair, and LSB==1 means use the upper byte of the pair). We can use an in-lane bytewise shuffle to do this.
  // We start with the identity shuffle (i.e., every byte stays in the same place), and then copy in the LSB
  // of the index. This has the effect of using the LSB to choose which of each pair of bytes to use.
  // Ternary logic is used to effect this bitwise blend.
  constexpr __v64qi shuffle_identity = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  };
  auto lsbSelector = _mm512_ternarylogic_epi32(__v64qi() + 0x1, shuffle_identity, v_shuf_idxs, 0xAC);

  // Use the selector to choose which byte to pick. We have to do this for the odd and even indexes independently
  // which means we end up doing some redundant work. Note that the second shuffle_epi8 only overwrites
  // the odd positions, accepting all the previous shuffle's even values.
  auto evenResult = _mm512_shuffle_epi8(evenPerm, lsbSelector);
  auto combinedResult = _mm512_mask_shuffle_epi8(evenResult, 0xAAAAAAAAAAAAAAAA, oddPerm, lsbSelector);

  return combinedResult;
}
#endif

#if defined(__AVX512F__)
inline target_overloads permutexvar {
  #if defined(__AVX512VBMI__)
    []<ymm_register<std::uint8_t> _Vec>(_Vec v, _Vec i)  { return _mm256_permutexvar_epi8(i.to_register(), v.to_register()); },
    []<zmm_register<std::uint8_t> _Vec>(_Vec v, _Vec i)  { return _mm512_permutexvar_epi8(i.to_register(), v.to_register()); },
  #else
    []<ymm_register<std::uint8_t> _Vec>(_Vec v, _Vec i)  { return __m256i(_mm512_permutex2var_epi8_emulated(__m512i(v.to_register()), __m512i(i.to_register()), __m512i())); },
    []<zmm_register<std::uint8_t> _Vec>(_Vec v, _Vec i)  { return _mm512_permutex2var_epi8_emulated(v.to_register(), i.to_register(), __m512i()); },
  #endif

    []<ymm_register<std::uint16_t> _Vec>(_Vec v, _Vec i) { return _mm256_permutexvar_epi16(i.to_register(), v.to_register()); },
    []<ymm_register<std::uint32_t> _Vec>(_Vec v, _Vec i) { return _mm256_permutexvar_epi32(i.to_register(), v.to_register()); },
    []<ymm_register<std::uint64_t> _Vec>(_Vec v, _Vec i) { return _mm256_permutexvar_epi64(i.to_register(), v.to_register()); },

    []<zmm_register<std::uint16_t> _Vec>(_Vec v, _Vec i) { return _mm512_permutexvar_epi16(i.to_register(), v.to_register()); },
    []<zmm_register<std::uint32_t> _Vec>(_Vec v, _Vec i) { return _mm512_permutexvar_epi32(i.to_register(), v.to_register()); },
    []<zmm_register<std::uint64_t> _Vec>(_Vec v, _Vec i) { return _mm512_permutexvar_epi64(i.to_register(), v.to_register()); },
};

inline target_overloads permutex2var {
  #if defined(__AVX512VBMI__)
    []<ymm_register<std::uint8_t> _Vec>(_Vec v0, _Vec i, _Vec v1)  { return _mm256_permutex2var_epi8(v0.to_register(), i.to_register(), v1.to_register()); },
    []<zmm_register<std::uint8_t> _Vec>(_Vec v0, _Vec i, _Vec v1)  { return _mm512_permutex2var_epi8(v0.to_register(), i.to_register(), v1.to_register()); },
  #else
    []<ymm_register<std::uint8_t> _Vec>(_Vec v0, _Vec i, _Vec v1)  { return __m256i(_mm512_permutex2var_epi8_emulated(__m512i(v0.to_register()), __m512i(i.to_register()), v1.to_register())); },
    []<zmm_register<std::uint8_t> _Vec>(_Vec v0, _Vec i, _Vec v1)  { return _mm512_permutex2var_epi8_emulated(v0.to_register(), i.to_register(), v1.to_register()); },
  #endif

    []<ymm_register<std::uint16_t> _Vec>(_Vec v0, _Vec i, _Vec v1) { return _mm256_permutex2var_epi16(v0.to_register(), i.to_register(), v1.to_register()); },
    []<ymm_register<std::uint32_t> _Vec>(_Vec v0, _Vec i, _Vec v1) { return _mm256_permutex2var_epi32(v0.to_register(), i.to_register(), v1.to_register()); },
    []<ymm_register<std::uint64_t> _Vec>(_Vec v0, _Vec i, _Vec v1) { return _mm256_permutex2var_epi64(v0.to_register(), i.to_register(), v1.to_register()); },

    []<zmm_register<std::uint16_t> _Vec>(_Vec v0, _Vec i, _Vec v1) { return _mm512_permutex2var_epi16(v0.to_register(), i.to_register(), v1.to_register()); },
    []<zmm_register<std::uint32_t> _Vec>(_Vec v0, _Vec i, _Vec v1) { return _mm512_permutex2var_epi32(v0.to_register(), i.to_register(), v1.to_register()); },
    []<zmm_register<std::uint64_t> _Vec>(_Vec v0, _Vec i, _Vec v1) { return _mm512_permutex2var_epi64(v0.to_register(), i.to_register(), v1.to_register()); },
};

#endif // AVX512F

/// Permute one register of Intel AVX-512 values.
template<std::unsigned_integral _Tp, typename _AbiT, std::integral _Ip, typename _AbiI>
requires (basic_vec<_Tp, _AbiT>::size <= vec<_Tp>::size) // One register's worth of values to permute. Note 
                                                         // that the indexes could be multiple registers.
inline auto permute_x86_register(x86_avx512_tag, const basic_vec<_Tp, _AbiT>& v, const basic_vec<_Ip, _AbiI>& indexes)
{
  const auto rv = permute<vec<_Tp>::size>(v, perm_uninitResize);

  auto impl = [=]<typename _Vec>(_Vec i) {
    // Permute instructions require the indexes to be the same size as the values being permute.
    const auto ri = permute<vec<_Tp>::size>(i, perm_uninitResize);
    auto t = vec<_Tp>(permutexvar(rv, rebind_cast<_Tp>(ri)));
    return fit_to_size<_Vec::size()>(t); // Permute instructions return a whole register.
  };

  return chunked_invoke(impl, indexes);
}

/// Permute two Intel AVX-512 registers of values (including 2 Intel AVX10 registers).
template<std::unsigned_integral _Tp, typename _AbiT, std::integral _Ip, typename _AbiI>
requires (basic_vec<_Tp, _AbiT>::size > vec<_Tp>::size &&
          basic_vec<_Tp, _AbiT>::size <= 2 * vec<_Tp>::size &&
          sizeof(_Tp) == sizeof(_Ip))
inline auto permute_x86_register(x86_avx512_tag, const basic_vec<_Tp, _AbiT>& v, const basic_vec<_Ip, _AbiI>& indexes)
{
  constexpr int nativeSize = vec<_Tp>::size;

  // Extract the two incoming registers and the indexes too as full size
  // registers. The registers can only be full size Intel AVX-512 since smaller
  // sizes would not fill two registers.
  const auto _lower = take<nativeSize>(v);
  const auto _upper = permute<nativeSize>(drop<nativeSize>(v), perm_uninitResize);

  // Resize the full size register back to the partial size. The intrinsics
  // require all of their registers to be the same size. We could size
  // everything to the common size (e.g., allowing a 256-bit permute on an
  // Intel AVX-512 machine) but that doesn't make much difference to the code
  // generated.
  auto impl = [=]<typename _Vec>(_Vec idx) {
    auto nativeIndexes = permute<nativeSize>(idx, perm_uninitResize);
    auto t = permutex2var(_lower, rebind_cast<_Tp>(nativeIndexes), _upper);
    return fit_to_size<_Vec::size>(vec<_Tp>(t));
  };

  return chunked_invoke(impl, indexes);
}

// Break a large set of values down into smaller groups which are permuted and
// combined. This acts recursively, repeatedly calling back into the permute to
// handle ever smaller pieces.
template<std::unsigned_integral _Tp, typename _AbiT, vec_integral _Ip>
inline auto permute_recurse(const basic_vec<_Tp, _AbiT>& values, const _Ip& indexes)
{
  constexpr auto nextPowerOf2 = std::bit_ceil(unsigned(basic_vec<_Tp, _AbiT>::size));
  constexpr auto halfSize = nextPowerOf2 / 2;

  // Split into two pieces, one a power-of-2, and the other the remainder.
  const auto [lowerValues, upperValues] = chunk<halfSize>(values);

  // Permute the lower and upper recursively. Note that the upper part could be
  // much smaller than the lower part (even a single element)  in which case it
  // would transform into code which did just enough work to permute that
  // smaller set of values. There is no need to replicate the entire full size
  // recursion on both parts.
  const auto indexedLower = permute(lowerValues, indexes);
  const auto indexedUpper = permute(upperValues, indexes);

  // Choose between lower and upper using the index bit for that power-of-2.
  return select((indexes & cw<halfSize>) == cw<0>, indexedLower, indexedUpper);
}

/// Entry point for the x86 permutations. This code tries to find the best
/// approach to permuting the supplied value, whether that is calling a dedicated
/// instruction, recursively breaking down large permutes into combinations of
/// smaller permutes, or transforming the permute into a permute of a different
/// element type to fit.
template<typename _Tp, typename _AbiTp, std::integral _Ip, typename _AbiIp, std::derived_from<x86_avx2_tag> VENDOR>
constexpr basic_vec<_Tp, _AbiIp>
permute(VENDOR tag, const basic_vec<_Tp, _AbiTp>& v, const basic_vec<_Ip, _AbiIp>& indexes)
{
  using _C = container_for_type<_Tp>;
  auto vc = simd_bit_cast<_C>(v);

  // Find the best match to do the permute.
  if constexpr (sizeof(_Tp) == 1 && sizeof(_Ip) == 1 &&
                basic_vec<_Tp, _AbiTp>::size <= 16 &&
                basic_vec<_Ip, _AbiIp>::size <= 16)
    // Special case - both source and index fit into one lane, so use an in-lane byte shuffle.
    return permute_byte_shuffle(v, indexes);
  else if constexpr (sizeof(_Tp) > 8)
    // Special case - large elements are dealt with as though they are pairs of smaller elements.
    return permute_large_elements(v, indexes);
  else if constexpr (std::same_as<VENDOR, x86_avx2_tag> && sizeof(_Tp) < 4)
  {
    // Special case - Intel AVX2 can't permute small element values so bump the size up to
    // something that can be supported.
    const auto r = permute(rebind_cast<uint32_t>(vc), indexes);
    const auto resizeBack = rebind_cast<_C>(r);
    return simd_bit_cast<_Tp>(resizeBack);
  }
  else if constexpr (sizeof(_Tp) != sizeof(_Ip))
    // Intel AVX ISAs have elements of the same size, so any mismatches of value and index must make the index size match.
    return permute(v, rebind_cast<container_for_type<_Tp>>(indexes));
  else if constexpr (requires {permute_x86_register(tag, vc, indexes); })
    // If a native register permute is available, use that to do the permute.
    // This must be done after the special cases (e.g., small elements on Intel AVX2)
    // are handled as it assumes that index and value are the same size.
     return simd_bit_cast<_Tp>(permute_x86_register(tag, vc, indexes));
  else
  {
    // Run out of options. Recursively break the problem down into smaller pieces which *can* be solved.
    return simd_bit_cast<_Tp>(permute_recurse(vc, indexes));
  }
}

// :TODO: Permute where contained in a single lane, or other special cases.
// duplicate a lane into multiple lanes to do multi-lane shuffle?

// :TODO: Special cases, like if the mask is less than 64-bits and the indexes
// has less than 128, and byte level permute is available (__AVX512VBMI2__),
// it will be a single instruction. Or it could use shuffle_epi8, and so on.
template<std::size_t _BytesTp, typename _AbiTp, typename _Ip, typename _AbiIp>
constexpr basic_mask<_BytesTp, _AbiIp>
permute(x86_avx512_tag, const basic_mask<_BytesTp, _AbiTp>& m, const basic_vec<_Ip, _AbiIp>& indexes)
{
  constexpr int _Np = basic_mask<_BytesTp, _AbiTp>::size;

  // Convert the mask to a vec with elements which are the same size as the
  // indexes, permute that, and then convert back to bits. :TODO: For ICX
  // onwards it may be faster to convert the indexes to bytes and then do a
  // permutex2var_epi8, which will allow a permute of up to 128 bits in the
  // mask.
  constexpr auto zeroMaskElement = vec<_Ip, _Np>();
  constexpr auto nonZeroMaskElement = vec<_Ip, _Np>(_Ip(~_Ip()));
  const auto maskAsSimd = select(mask<_Ip, _Np>(m), nonZeroMaskElement, zeroMaskElement);

  const auto r = permute(maskAsSimd, indexes) != cw<0>;
  return  basic_mask<_BytesTp, _AbiIp>(r);
}

/// Use a generator to permute a compact bit mask.
template<simd_size_type _OutSize, std::size_t _Bytes, typename _Abi, typename _Gp>
constexpr auto
generated_permute(x86_avx512_tag, const basic_mask<_Bytes, _Abi>& m, _Gp)
{
#if defined(__AVX512BMI__)
  using _I = uint8_t;
#else
  // Not always a good idea to use 16-bit, as the instruction itself is quite
  // slow, but using 32-bit could lead to multiple registers which is also slow.
  // One special case is when there are fewer bits than a lane has bytes, in
  // which case epi8 could be used instead.
  using _I = std::conditional_t<basic_mask<_Bytes, _Abi>::size <= 16, uint8_t, uint16_t>;
#endif

  using _R = vec<_I, _OutSize>; 

  // :TODO: Provide constexpr version which permutes the bits directly.

  // generate a set of indexes to do the permutation, and then use a real indexed permute with those.
  constexpr auto indexes = [=]{
    if constexpr (index_generator_function_with_size<_Gp>)
      return generate<_R>([](auto idx) -> _I { return _I(_Gp{}(idx, _OutSize)); });
    else
      return generate<_R>([](auto idx) -> _I { return _I(_Gp{}(idx)); });
  }();
  auto p = permute(m, indexes);

  // Some of the indexes might be `zero_element` which need to be zeroed
  // out. Create a suitable mask for them. :TODO: Once constexpr operator== is
  // available this can be replaced by `indexes == zero_element`
  using OutputMask = resize_t<_OutSize, basic_mask<_Bytes, _Abi>>;
  constexpr auto notZeroMask = [=]{
    if constexpr (index_generator_function_with_size<_Gp>)
      return OutputMask([=](auto i) { return _Gp{}(i, _OutSize) != zero_element; });
    else
      return OutputMask([=](auto i) { return _Gp{}(i) != zero_element; });
  }();

  // Unconditionally mask the zero elements as the compiler will elide this
  // where it is unnecessary.
  return select(notZeroMask, p, OutputMask());
}

// :TODO: if the mask has no more than 8 bits then use bitshuffle_epi64_mask

#if defined(__AVX512F__)
inline target_overloads compress_one_register {
#if defined(__AVX512VBMI2__)
    // __AVX512VBMI2__ has 8 and 16-bit compress instructions.

    // int 8
    []<xmm_register<std::uint8_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm_mask_compress_epi8(f.to_register(), m.to_register(), v.to_register())); },
    []<ymm_register<std::uint8_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm256_mask_compress_epi8(f.to_register(), m.to_register(), v.to_register())); },
    []<zmm_register<std::uint8_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm512_mask_compress_epi8(f.to_register(), m.to_register(), v.to_register())); },

  // int 16
    []<xmm_register<std::uint16_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm_mask_compress_epi16(f.to_register(), m.to_register(), v.to_register())); },
    []<ymm_register<std::uint16_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm256_mask_compress_epi16(f.to_register(), m.to_register(), v.to_register())); },
    []<zmm_register<std::uint16_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm512_mask_compress_epi16(f.to_register(), m.to_register(), v.to_register())); },
#else
    // In the absence of __AVX512VBMI2__ instructions for 8/16-bit, synthesise them by using their 32-bit counterparts.

    []<xmm_register<uint8_t> _Vec>(_Vec v, auto m, auto f) {
      auto compressed = _mm512_mask_compress_epi32(_mm512_cvtepi8_epi32(f.to_register()),
                                                   m.to_register(),
                                                   _mm512_cvtepi8_epi32(v.to_register()));
      return _Vec(_mm512_cvtepi32_epi8(compressed));
    },
    []<xmm_register<uint16_t> _Vec>(_Vec v, auto m, auto f) {
      auto compressed = _mm256_mask_compress_epi32(_mm256_cvtepi16_epi32(f.to_register()),
                                                   m.to_register(),
                                                   _mm256_cvtepi16_epi32(v.to_register()));
      return _Vec(_mm256_cvtepi32_epi16(compressed));
    },
    []<ymm_register<uint16_t> _Vec>(_Vec v, auto m, auto f) {
      auto compressed = _mm512_mask_compress_epi32(_mm512_cvtepi16_epi32(f.to_register()),
                                                   m.to_register(),
                                                   _mm512_cvtepi16_epi32(v.to_register()));
      return _Vec(_mm512_cvtepi32_epi16(compressed));
    },

#endif

    // int 32
    []<xmm_register<std::uint32_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm_mask_compress_epi32(f.to_register(), m.to_register(), v.to_register())); },
    []<ymm_register<std::uint32_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm256_mask_compress_epi32(f.to_register(), m.to_register(), v.to_register())); },
    []<zmm_register<std::uint32_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm512_mask_compress_epi32(f.to_register(), m.to_register(), v.to_register())); },

    // int 64
    []<xmm_register<std::uint64_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm_mask_compress_epi64(f.to_register(), m.to_register(), v.to_register())); },
    []<ymm_register<std::uint64_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm256_mask_compress_epi64(f.to_register(), m.to_register(), v.to_register())); },
    []<zmm_register<std::uint64_t> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm512_mask_compress_epi64(f.to_register(), m.to_register(), v.to_register())); },

    // int 128 - Double up each bit to allow 64-bit compress to be used to handle 128-bit elements as pairs of bits.
    []<xmm_register<unsigned __int128> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm_mask_compress_epi64(f.to_register(), __mmask8(dupBits(m.to_register())), v.to_register())); },
    []<ymm_register<unsigned __int128> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm256_mask_compress_epi64(f.to_register(), __mmask8(dupBits(m.to_register())), v.to_register())); },
    []<zmm_register<unsigned __int128> _Vec>(_Vec v, auto m, auto f)
      { return _Vec(_mm512_mask_compress_epi64(f.to_register(), __mmask8(dupBits(m.to_register())), v.to_register())); },

    [](auto unhandled, auto, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled;}
  };


  template <typename _Tp, typename _Abi, typename VENDOR>
  constexpr void
  compress_by_mask_to_memory(VENDOR, _Tp* ptr, const typename basic_vec<_Tp, _Abi>::mask_type& selector,
                             const basic_vec<_Tp, _Abi>& values, _Tp fill_value)
  {
    // Given a block, compress each bit of the block into consecutive memory regions.
    auto nextPtr = ptr;
    auto compressBlock = [&]<typename _Vec>(_Vec x, typename _Vec::mask_type m)
    {
      const auto fillV = _Vec(fill_value);
      auto compressed = compress_one_register(x, m, fillV);
      unchecked_store(compressed, nextPtr, _Vec::size);
      nextPtr += reduce_count(m);
    };

    // Break the vec into pieces no bigger than 16 elements so that the compression operation can be done
    // entirely within a single ZMM register.
    constexpr int numBlocks = (sizeof(_Tp) >= 4 || std::derived_from<VENDOR, x86_avxsnc_tag>) ? vec<_Tp>::size : 16;
    chunked_invoke<numBlocks>(compressBlock, values, selector);
  }

  // Special case for compression on __AVX512VBMI2__ in which a set of 8-bit indexes is
  // compressed, and then the indexes are used in a permute.
  template<vec_type _V>
constexpr _V
compress_by_mask_snc(const _V& values, const typename _V::mask_type& selector, typename _V::value_type fill_value)
{
  constexpr int _Np = _V::size;
  static_assert(_Np <= detail::maxBytesInVec, "Cannot compress more than one native register at once");

  // Use a byte compression to turn the entire 64-bit mask into a set of 64
  // indexes, and then use that to permute the input. The output needs to be
  // masked to introduce zeros in unwritten elements.
  constexpr auto indexes = iota<vec<std::uint8_t, _Np>>;
  const auto compressedIndexes = chunked_invoke(compress_one_register, indexes, mask<uint8_t, _Np>(selector), vec<uint8_t, _Np>());
  auto compressed = permute(values, compressedIndexes);
  auto numSetBits = reduce_count(selector);
  auto finalMask = _V::mask_type::__mask_from_count(numSetBits);
  return select(finalMask, compressed, _V(fill_value));
}

/// Compress a set of vec elements by the supplied mask using AVX512F. Note that the fill value is
/// unconditionally a _Tp, unlike the generic version where it defaults to uninitialised. This is because
/// setting a fill value in Intel AVX-512 makes no difference to performance.
template<vec_type _V, std::derived_from<x86_avx512_tag> VENDOR>
constexpr _V
compress_by_mask(VENDOR tag, const _V& values, const typename _V::mask_type& selector,
                 typename _V::value_type fill_value = {})
{
  constexpr int _Np = _V::size;
  using _Tp = typename _V::value_type;
  static_assert(vec_container_element<_Tp>, "Caller must convert to container");

  // Work out the maximum number of expansion elements permitted by the processor for this target.
  constexpr bool isSnc = std::derived_from<VENDOR, x86_avxsnc_tag>;
  constexpr int maxElementsPerExpand = isSnc ? vec<_Tp>::size : std::min<int>(16, vec<_Tp>::size);

  if constexpr (_Np <= maxElementsPerExpand)
  {
    // If the number of elements fits into what can be expanded in one
    // instruction (even allowing for using larger granularity), then do it.
    return compress_one_register(values, selector, _V(fill_value));
  }
  else if constexpr (isSnc && _Np <= detail::maxBytesInVec)
    // Special case for __AVX512VBMI2__ where the number of elements is 64 or fewer.
    return compress_by_mask_snc(values, selector, fill_value);
  else
  {
    // No other option - compress directly into memory.
    _V result(fill_value);
    compress_by_mask_to_memory(tag, reinterpret_cast<_Tp*>(&result), selector, values, fill_value);
    return result;
  }
}
#endif // AVX512F

#if defined(__BMI2__)
template<mask_type _M>
constexpr _M
compress_mask_by_mask(x86_avx2_tag, const _M& values, const _M& selector)
{
  constexpr int _Np = _M::size;

  // At time of implementation only 128 bits were supported in masks. If this changes we need to
  // modify the code below to handle more chunks of data.
  static_assert(_Np <= 128);

  using _R = std::conditional_t<(_Np > 64), __uint128_t, uint64_t>;

  // Process up to the first 64 bits.
  constexpr int lowerN = std::min<int>(_Np, 64);
  auto maskAsUll = detail::fit_to_size<lowerN>(selector).to_ullong();
  _R resultInt = _pext_u64(detail::fit_to_size<lowerN>(values).to_ullong(), maskAsUll);

  // Process any upper bits.
  if constexpr (_Np > 64)
  {
    _R upperCompressed = _pext_u64(drop<64>(values).to_ullong(), drop<64>(selector).to_ullong());
    resultInt |= (upperCompressed << std::popcount((uint64_t)maskAsUll));
  }

  // The result is in an integer. Convert that back into a mask.
  _M resultMask((uint64_t)resultInt);
  if constexpr (_Np > 64) {
    auto upper = resize_t<_Np - 64, _M>(uint64_t(resultInt >> 64));
    resultMask = cat(take<64>(resultMask), upper);
  }

  return resultMask;
}

template<mask_type _M>
constexpr _M
compress_mask_by_mask(x86_avx512_tag, const _M& v, const _M& selector)
{

  constexpr int _Np = _M::size;

  auto vd = v.to_builtin();
  auto md = selector.to_builtin();

  // At time of implementation only 128 bits were supported in masks. If this changes we need to
  // modify the code below to handle more chunks of data.
  static_assert(_Np <= 128);

  // Process the first 64 bits.
  using _B = typename _M::builtin_type;
  _B result = _pext_u64((uint64_t)vd, (uint64_t)md);

  // Process the top 64-bits too and then combine them together.
  if constexpr (_Np > 64)
  {
    _B upperCompressed = _pext_u64(vd >> 64, md >> 64);
    result |= (upperCompressed << std::popcount((uint64_t)md));
  }

  return _M::from_builtin(result);
}
#endif

#if defined(__AVX512F__)
inline target_overloads expand_one_register {
#if defined(__AVX512VBMI2__)
  // int 8
  []<xmm_register<std::uint8_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm_mask_expand_epi8(alt.to_register(), m.to_register(), v.to_register())); },
  []<ymm_register<std::uint8_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm256_mask_expand_epi8(alt.to_register(), m.to_register(), v.to_register())); },
  []<zmm_register<std::uint8_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm512_mask_expand_epi8(alt.to_register(), m.to_register(), v.to_register())); },

  []<xmm_register<std::uint16_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm_mask_expand_epi16(alt.to_register(), m.to_register(), v.to_register())); },
  []<ymm_register<std::uint16_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm256_mask_expand_epi16(alt.to_register(), m.to_register(), v.to_register())); },
  []<zmm_register<std::uint16_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm512_mask_expand_epi16(alt.to_register(), m.to_register(), v.to_register())); },
#else

  // When VBMI2 is not available, small types can only be expanded by
  // converting to a 4B type, expanding that, and converting back again. For the
  // expand instruction to have only one register to work it must accept no
  // more than 16 elements to start with (i.e., 128 or 256-bits respectively).
  []<xmm_register<uint8_t> _Vec>(_Vec v, auto m, auto alt) {
    auto compressed = _mm512_maskz_expand_epi32(m.to_register(),
                                                _mm512_cvtepi8_epi32(v.to_register()));
    return _Vec(_mm512_mask_cvtepi32_epi8(alt.to_register(), m.to_register(), compressed));
  },
  []<xmm_register<uint16_t> _Vec>(_Vec v, auto m, auto alt) {
    auto compressed = _mm256_maskz_expand_epi32(m.to_register(),
                                                _mm256_cvtepi16_epi32(v.to_register()));
    return _Vec(_mm256_mask_cvtepi32_epi16(alt.to_register(), m.to_register(), compressed));
  },
  []<ymm_register<uint16_t> _Vec>(_Vec v, auto m, auto alt) {
    auto compressed = _mm512_maskz_expand_epi32(m.to_register(),
                                                _mm512_cvtepi16_epi32(v.to_register()));
    return _Vec(_mm512_mask_cvtepi32_epi16(alt.to_register(), m.to_register(), compressed));
  },
#endif

  // 32-bit
  []<xmm_register<std::uint32_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm_mask_expand_epi32(alt.to_register(), m.to_register(), v.to_register())); },
  []<ymm_register<std::uint32_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm256_mask_expand_epi32(alt.to_register(), m.to_register(), v.to_register())); },
  []<zmm_register<std::uint32_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm512_mask_expand_epi32(alt.to_register(), m.to_register(), v.to_register())); },

  // 64-bit
  []<xmm_register<std::uint64_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm_mask_expand_epi64(alt.to_register(), m.to_register(), v.to_register())); },
  []<ymm_register<std::uint64_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm256_mask_expand_epi64(alt.to_register(), m.to_register(), v.to_register())); },
  []<zmm_register<std::uint64_t> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm512_mask_expand_epi64(alt.to_register(), m.to_register(), v.to_register())); },

  // int 128 - Double up each bit to allow 64-bit compress to be used to handle 128-bit elements as pairs of bits.
  []<xmm_register<unsigned __int128> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm_mask_expand_epi64(alt.to_register(), __mmask8(dupBits(m.to_register())), v.to_register())); },
  []<ymm_register<unsigned __int128> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm256_mask_expand_epi64(alt.to_register(), __mmask8(dupBits(m.to_register())), v.to_register())); },
  []<zmm_register<unsigned __int128> _Vec>(_Vec v, auto m, auto alt)
    { return _Vec(_mm512_mask_expand_epi64(alt.to_register(), __mmask8(dupBits(m.to_register())), v.to_register())); },

  [](auto unhandled, auto, auto) { static_assert(dependent_false<decltype(unhandled)>, "No target intrinsics"); return unhandled;}
};

template<vec_type _V, typename VENDOR>
constexpr _V
expand_by_mask_from_memory(VENDOR, const typename _V::value_type* ptr,
                           const typename _V::mask_type& selector, const _V& original)
{
  using _Tp = typename _V::value_type;

  // Given a block, expand the indexes and move the output offset.
  auto nextPos = ptr;
  auto expandBlock = [&]<typename _Vec>(_Vec x, typename _Vec::mask_type m) {
    auto t = detail::load<_Vec>(target, std::span<const _Tp, _Vec::size>(nextPos, _Vec::size), detail::add_unchecked_flag(flags<>{}));
    nextPos += reduce_count(m);
    return expand_one_register(t, m, x);
  };

  // Break the vec into pieces no bigger than 16 elements so that the expansion operation can be done
  // entirely within a single ZMM register.
  constexpr int numBlocks = (sizeof(_Tp) >= 4 || std::derived_from<VENDOR, x86_avxsnc_tag>) ? vec<_Tp>::size : 16;
  return chunked_invoke<numBlocks>(expandBlock, original, selector);
}

template<vec_type _V>
constexpr _V
expand_by_mask_snc(const _V& value, const typename _V::mask_type& selector, const _V& original)
{
  constexpr int _Np = _V::size;

  static_assert(_Np <= detail::maxBytesInVec);

  // __AVX512VBMI2__ sizes greater than 64 values can be expanded by converting the mask into indexes
  // by expanding iota, and then permuting using the indexes.
  // :TODO: The permute should cope better with mixed sizes than it currently does.
  constexpr auto indexes = iota<vec<std::uint8_t, _Np>>;
  const auto expandedIndexes = chunked_invoke(expand_one_register, indexes, mask<uint8_t, _Np>(selector), vec<uint8_t, _Np>());
  auto expanded = permute(value, expandedIndexes);
  return select(selector, expanded, original);
}

template<vec_type _V, std::derived_from<x86_avx512_tag> VENDOR>
constexpr _V
expand_by_mask(VENDOR tag, const _V& values, const typename _V::mask_type& selector, const _V& original)
{
  using _Tp = typename _V::value_type;
  static_assert(vec_container_element<_Tp>, "Caller must convert to container");
  constexpr auto _Np = _V::size;

  // Work out the maximum number of expansion elements permitted by the processor for this target.
  constexpr bool isSnc = std::derived_from<VENDOR, x86_avxsnc_tag>;
  const int maxElementsPerExpand = isSnc ? vec<_Tp>::size() : std::min<int>(16, vec<_Tp>::size());

  if constexpr (_Np <= maxElementsPerExpand)
    // If the number of elements fits into what can be expanded in one
    // instruction (even allowing for using larger granularity), then do it.
    return expand_one_register(values, selector, original);
  else if constexpr (isSnc && _Np <= detail::maxBytesInVec)
    // Special case for __AVX512VBMI2__ where the number of elements is less than 64.
    return expand_by_mask_snc(values, selector, original);
  else
  {
    // No other option - load from memory.
    return expand_by_mask_from_memory(tag, reinterpret_cast<const _Tp*>(&values), selector, original);
  }
}
#endif // AVX512F

#if defined(__BMI2__)
template<mask_type _M>
constexpr _M
expand_mask_by_mask(x86_avx_tag, const _M& values, const _M& selector, const _M& original)
{
  constexpr int _Np = _M::size;

  // At time of implementation only 128 bits were supported in masks. If this changes we need to
  // modify the code below to handle more chunks of data.
  static_assert(_Np <= 128);

  using _R = std::conditional_t<(_Np > 64), __uint128_t, uint64_t>;

  // Process the first 64 bits.
  constexpr int lowerN = std::min<int>(_Np, 64);
  auto lowerSelector = detail::fit_to_size<lowerN>(selector).to_ullong();
  auto lowerValue = detail::fit_to_size<lowerN>(values).to_ullong();
  _R resultInt = _pdep_u64(lowerValue, lowerSelector);

  // Merge in any upper bits for large masks.
  if constexpr (_Np > 64)
  {
    _R upper64Value = drop<64>(values).to_ullong();
    uint64_t upper64Selector = drop<64>(selector).to_ullong();
    auto wholeValue = (upper64Value << 64) | _R(lowerValue);
    auto bitsToExpand = uint64_t(wholeValue >> std::popcount(lowerSelector));
    _R expandedBits = _pdep_u64(bitsToExpand, upper64Selector);
    resultInt |= _R(expandedBits) << 64;
  }

  _M resultMask((uint64_t)resultInt);
  if constexpr (_Np > 64)
    resultMask = cat(take<64>(resultMask), resize_t<_Np - 64, _M>(uint64_t(resultInt >> 64)));

  // Blend in any original bits and return the final result.
  auto final = resultMask.to_builtin() | (original.to_builtin() & ~(selector.to_builtin()));
  return _M::from_builtin(final);
}

template<mask_type _M>
constexpr _M
expand_mask_by_mask(x86_avx512_tag, const _M& values, const _M& selector, const _M& original)
{
  constexpr int _Np = _M::size;

  auto vd = values.to_builtin();
  auto sd = selector.to_builtin();

  // At time of implementation only 128 bits were supported in masks. If this changes we need to
  // modify the code below to handle more chunks of data.
  static_assert(_Np <= 128);

  // Process the first 64 bits.
  typename _M::builtin_type result = _pdep_u64((uint64_t)vd, (uint64_t)sd);

  // // If there are more than 64-bits, process the top 64-bits too and then combine them together.
  if constexpr (_Np > 64)
  {
    uint64_t useBits = (uint64_t)(vd >> std::popcount((uint64_t)sd));
    typename _M::builtin_type expandedBits = _pdep_u64(useBits, (uint64_t)(sd >> 64));
    result |= expandedBits << 64;
  }

  // Blend in the original data in unmasked positions.
  return _M::from_builtin(result | (original.to_builtin() & ~sd));
}
#endif

/// Custom x86 versions of some of the named permutes. Typically these
/// optimized for Intel AVX-512 masks, since the compiler can do a good job of most
/// vec value permutes without help.

/// @brief A named permute which discards the first _Np values of a vec, returning the remainder.
template <int _Np, mask_type _Mp>
constexpr auto permute_drop(x86_avx512_tag, const _Mp& in) noexcept
{
  static_assert(_Mp::size >= _Np, "Too short to drop N elements");
  return resize_t<_Mp::size - _Np, _Mp>::from_builtin(in.to_register() >> _Np);
}

/// Create a mask containing only the first _Np elements. The remaining elements are discarded.
template <int _Np, mask_type _Mp>
constexpr auto permute_take(x86_avx512_tag, const _Mp& in) noexcept
{
  static_assert(_Np <= _Mp::size, "Too short to drop N elements");
  return resize_t<_Np, _Mp>::from_builtin(in.to_register());
}

/// @brief Make the given mask grow to _Np elements, filling the extra elements with false.
template<int _Np, mask_type _Mp>
constexpr auto permute_grow(x86_avx512_tag, const _Mp& in) noexcept {
  static_assert(_Np >= _Mp::size, "Must grow to a size bigger than the input");
  return resize_t<_Np, _Mp>::from_builtin(in.to_register());
}

/// Make the given mask grow to _Np elements, filling the extra elements with the supplied value.
template<int _Np, mask_type _Mp>
constexpr auto permute_grow(x86_avx512_tag, const _Mp& in, bool fill) noexcept {
  static_assert(_Np >= _Mp::size, "Must grow to a size bigger than the input");
  using _NewMp = resize_t<_Np, _Mp>;
  using _CM = typename _NewMp::builtin_type;

  auto f = fill ? ~(_CM()) : _CM();
  const auto lowerMask = ~(_CM()) << _Mp::size();

  return _NewMp::from_builtin(in.to_register() | (f & lowerMask));
}

/// Utility function to create a bit mask containing bits are set at a distance apart of _Stride.
/// @tparam _Mp The output mask type
/// @tparam _Stride The distance between one set bit and the next
/// @return An integer containing _Mp::size bits set at a distance of _Stride apart.
template<mask_type _Mp, int _Stride>
constexpr auto every_nth_bit() {
  using _Out = resize_t<_Mp::size * _Stride, _Mp>;
  using _I = typename _Out::builtin_type;
  return [=]<std::size_t... _Iota>(std::index_sequence<_Iota...>) {
      return ((_I(1) << (_Iota * _Stride)) | ...);
  }(std::make_index_sequence<_Mp::size>{});
}

#if defined(__AVX512F__)
/// Extract every _Np'th element of the input mask.
template<int _Np, mask_type _Mp>
constexpr auto permute_stride(x86_avx512_tag, const _Mp& in) noexcept {
  constexpr auto _OldSize = _Mp::size;

  // If the mask size increases this code will need more work.
  static_assert(_OldSize <= 128, "Hard-coded to a pair of 64-bit pext instructions");

  // Easy special case.
  if constexpr (_Np == 1) return in;

  constexpr int _NewSize = (_OldSize + (_Np - 1)) / _Np;
  using _NewMp = resize_t<_NewSize, _Mp>;

  auto iv = in.to_register();

  // Handle the lower 64-bits. Note that the output will never exceed 64 bits,
  // since stride of 1 is handled above, and the maximum size of a mask is
  // currently 128-bits, which means a stride of 2 will never generate more than
  // 64 bits.
  constexpr auto bits = every_nth_bit<_NewMp, _Np>();
  uint64_t t = _pext_u64((uint64_t)iv, (uint64_t)bits);

  // Handle the upper 64-bits if they are also present.
  if constexpr (_OldSize > 64) {
    auto upperBits = uint64_t(bits >> 64);
    uint64_t b = _pext_u64((uint64_t)(iv >> 64), upperBits);
    t |= b << std::popcount(uint64_t(bits & 0xFFFFFFFFFFFFFFFF));
  }

  return _NewMp::from_builtin(t);
}
#endif

} // detail namespace

} // namespace _XVEC_NAMESPACE::simd
