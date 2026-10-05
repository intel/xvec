# Release Notes

This page documents the version history and changes for the **xvec** library.

## xvec

### Unreleased

- Initial open-source availability, copied over from a previously private internal version of the library.
- Fixed issue which prevented the scatter/gather functions from taking signed indexes. All variants of these functions now take integral indexes, as per the C++26 working draft.
- Fixed issue with generator constructors passing the index value as the wrong type. The index is now `simd_size_type` instead of the previously incorrect `size_t` type.
