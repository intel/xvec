# Release Notes

This page documents the version history and changes for the **xvec** library.

## xvec

### Unreleased

- Initial open-source availability, copied over from a previously private internal version of the library.
- Fixed issue which prevented the scatter/gather functions from taking signed indexes. All variants of these functions now take integral indexes, as per the C++26 working draft.
- Fixed issue with generator constructors passing the index value as the wrong type. The index is now `simd_size_type` instead of the previously incorrect `size_t` type.
- Added best-effort freestanding support on x86. An optimized `-ffreestanding` probe checks that representative operations, including arithmetic on a user-defined complex type, produce no external symbol references. This does not guarantee that every operation is runtime-free. Forced-scalar freestanding builds are not supported.
- Added `xvec::simd::x86::dpphps(a, b, acc)`, a width-independent wrapper for the Intel AVX10 `VDPPHPS` instruction. Each FP16 input must have exactly two elements per FP32 accumulator element. Callers choose how to handle an unpaired element.
- Added `inner_product(lhs, rhs, init)` to compute the inner product of FP16 vectors of any size. The FP16 inputs are widened and accumulated in FP32 to improve precision. It uses the Intel AVX10 FP16 dot-product instruction on supported targets, or falls back to a generic version for older targets.
