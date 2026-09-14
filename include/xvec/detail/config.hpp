//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header is included first by every xvec header, so it must not include
// anything from xvec. Only definitions with no dependencies belong here.

/// The enclosing namespace for the whole library. Every header opens its
/// namespaces through this macro rather than naming `xvec` directly, so that the
/// library can be relocated wholesale.
///
/// The library lives in `xvec::simd` by default, mirroring the `std::simd`
/// structure of the C++26 working draft. Define `_XVEC_NAMESPACE` before
/// including any xvec header to place it elsewhere - most usefully `std`, for
/// when this implementation is absorbed into a standard library.
#ifndef _XVEC_NAMESPACE
#define _XVEC_NAMESPACE xvec
#endif
