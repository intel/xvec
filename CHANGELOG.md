# Release Notes

This page documents the version history and changes for the **xvec** library.

## xvec

### Unreleased

- Initial open-source availability, copied over from a previously private internal version of the library.
- Fixed issue which prevented the scatter/gather functions from taking signed indexes. All variants of these functions now take integral indexes, as per the C++26 working draft.
- Fixed issue with generator constructors passing the index value as the wrong type. The index is now `simd_size_type` instead of the previously incorrect `size_t` type.
- Added freestanding mode (`-ffreestanding`) to allow the library to be used in environments with no runtime support. This disables iostream operations, maths library functions, and complex number support. Note that complex-like types can be supported even in the absence of `std::complex` by defining an appropriate `complex_trait` specialisation.
