# Extensible Vectorization Library - xvec

[![Documentation](https://github.com/intel/xvec/actions/workflows/docs.yml/badge.svg)](https://github.com/intel/xvec/actions/workflows/docs.yml)
[![SmokeTest verification](https://github.com/intel/xvec/actions/workflows/smoke.yml/badge.svg)](https://github.com/intel/xvec/actions/workflows/smoke.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0_WITH_LLVM--exception-blue.svg)](LICENSE.txt)

xvec is a header-only implementation of the C++26 data-parallel library,
commonly known as `std::simd`. It provides vector and mask types for expressing
portable SIMD operations without fixing an algorithm to a particular register
width or instruction set.

The library supports native and fixed-size vectors, including non-power-of-two
sizes. Its facilities include arithmetic, comparisons, masks, mathematical
functions, reductions, memory operations, permutations, complex numbers, and
user-defined vectorizable types. The C++26 API is provided in the
`xvec::simd` namespace, corresponding to `std::simd`.

Most SIMD operations are delegated to the compiler. This allows xvec to support
targets for which the compiler provides suitable vector operations. The library
also contains target-specific implementations where these improve performance
or provide operations that cannot be expressed efficiently through generic
compiler support. Optimized support is currently provided for x86.

xvec also provides portable and target-specific extensions beyond C++26. Some
of these extensions are being proposed for C++29.

> [!IMPORTANT]
> xvec is suitable for production use, but its API and ABI continue to evolve.
> Changes to the C++ standard, experience from use, and feedback from users may
> require compatibility-breaking changes.

## Features

- Native and fixed-size SIMD vectors.
- Vector sizes that do not need to be powers of two.
- Arithmetic, bitwise, comparison, mathematical, and reduction operations.
- Masks, selection, and predicated operations.
- Loads, stores, gathers, and scatters.
- Compile-time and runtime permutations.
- Complex and user-defined element types.
- `constexpr` support where supported by the language and compiler.
- Generic implementations based on compiler vector operations.
- Optimized implementations and extensions for x86.
- Standards-track, portable xvec, and target extensions beyond C++26.

## Requirements

xvec requires C++20 or later.

The oldest compiler versions covered by regression testing are:

| Compiler | Oldest tested version |
| --- | ---: |
| GCC | 13 |
| Clang | 17 |
| oneAPI DPC++/C++ Compiler | 2024.0 |

Older compiler versions may also work but are not included in regression
testing.

## Getting started

Add the repository's `include` directory to the compiler include path and
include the top-level header:

```cpp
#include <xvec/simd>
```

For example:

```cpp
#include <xvec/simd>

int main()
{
    using xvec::simd::vec;

    const vec<float, 4> values(
        [](auto index) { return static_cast<float>(index + 1); });

    const auto result = values * vec<float, 4>(2.0f);

    return result[0] == 2.0f && result[3] == 8.0f ? 0 : 1;
}
```

Compile the example by enabling C++20 and adding the xvec headers to the include
path:

```bash
g++ -std=c++20 -O2 -I/path/to/xvec/include example.cpp
```

Use the compiler's normal target options to select the desired architecture.
For example, `-march=native` can be used for a local build. Code intended for
distribution should select an appropriate baseline target instead.

## C++26 compatibility

The xvec API follows the namespace and type structure of the C++26
data-parallel library:

| C++26 | xvec |
| --- | --- |
| `std::simd::vec<T, N>` | `xvec::simd::vec<T, N>` |
| `std::simd::mask<T, N>` | `xvec::simd::mask<T, N>` |

This correspondence allows code to be adapted between xvec and a conforming
standard library implementation without changing its underlying programming
model.

The C++26 interface is described in:

- [C++ working draft: Data-parallel types](https://eel.is/c++draft/simd)
- [cppreference: `std::simd`](https://en.cppreference.com/w/cpp/numeric/simd)

## Extensions beyond C++26

xvec provides functionality beyond the C++26 data-parallel library. The
documentation classifies these APIs according to their scope and relationship
to the C++ standard:

- **Standards-track extensions** are being proposed for a future C++ standard,
  currently C++29. Their interfaces may change as the proposals progress
  through standardization.
- **Portable xvec extensions** are not part of C++26 but are designed to work
  across targets using the generic compiler-backed implementation.
- **Target extensions** expose operations or properties associated with a
  particular target family. The current target extensions are provided under
  `xvec::simd::x86`.

These extensions include additional permutation and data-movement operations,
specialized arithmetic, generic SIMD algorithms, user-defined type support,
and target-specific operations and concepts.

API stability is separate from an extension's category:

- **Stable** APIs are intended for continued use but remain subject to the
  project's compatibility policy.
- **Experimental** APIs are still being evaluated and are more likely to change
  or be removed.
- **Deprecated** APIs are retained temporarily to support migration.

Extensions, including experimental APIs, are currently always available. A
future release is expected to provide a mechanism for explicitly enabling
experimental functionality. Code that depends on experimental APIs should not
assume source or ABI stability.

## Target support

xvec expresses most operations using compiler vector types and operations. A
target can therefore use the generic implementation when its compiler provides
the required SIMD support.

The performance of the generic implementation depends on the compiler and
target. A target-specific compatibility layer is not required, but can be added
to:

- Improve generated code for common operations.
- Use instructions that are not exposed efficiently through generic compiler
  vector operations.
- Optimize permutations, reductions, masks, gathers, and scatters.
- Work around target-specific compiler limitations.
- Provide target-specific extensions.

xvec currently includes a target-specific compatibility layer for x86. Other
targets can begin with the generic implementation and add specialized
implementations where useful.

Documentation for evaluating and adding support for new targets will be added
as the porting process is developed.

## Documentation

The full documentation includes the API reference, guides, and worked examples:

**https://intel.github.io/xvec/**

The documentation is generated from the source using Doxygen and published from
the `main` branch.

For the containerized development environment, GHCR toolchain image, and VS Code
Dev Containers workflow, see
[docs/development-container.md](docs/development-container.md).

## Releases and compatibility

The `main` branch contains the latest development version. It is continuously
tested but may include API changes as the C++ SIMD specification and xvec
extensions evolve.

Versioned snapshots are published through
[GitHub Releases](https://github.com/intel/xvec/releases). Downstream projects
that require a reproducible dependency should use a tagged release rather than
an arbitrary commit from `main`.

Release notes identify:

- New functionality.
- Breaking changes.
- Deprecations and removals.
- Correctness and performance fixes.
- Compiler and target support changes.
- Known limitations and migration guidance.

## Contributing

Contributions, bug reports, feature requests, and other feedback are welcome.

- Report bugs or request features through
  [GitHub Issues](https://github.com/intel/xvec/issues).
- Propose changes through
  [GitHub Pull Requests](https://github.com/intel/xvec/pulls).
- Do not report suspected security vulnerabilities in a public issue.

Contributions may improve generic compiler-based support or add optimized
support for new targets. Target-specific implementations should preserve the
portable public API and include appropriate correctness and performance tests.

## Security

See the [Intel Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability privately rather than through a public GitHub issue. You may also review the repository [Security Policy](SECURITY.md).

## License

xvec is licensed under the
[Apache License 2.0 with LLVM Exceptions](LICENSE.txt).

See the repository's `LICENSE.txt` file for the complete terms, and
[third-party-programs.txt](third-party-programs.txt) for third party software
referenced by this project.
