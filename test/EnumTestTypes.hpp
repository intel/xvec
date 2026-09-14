//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "SimdTestUtilities.hpp"

// Uses both positive and negative values.
enum UnscopedEnumTest {
  UE_MINUS_ZERO = 0, UE_MINUS_ONE = -1, UE_MINUS_TWO = -2, UE_MINUS_THREE = -3, UE_MINUS_FOUR = -4, UE_MINUS_FIVE = -5,
  UE_MINUS_SIZE = -6, UE_MINUS_SEVEN = -7, UE_MINUS_EIGHT = -8, UE_MINUS_NINE = -9,
  UE_ZERO = 0, UE_ONE = 1, UE_TWO = 2, UE_THREE = 3, UE_FOUR = 4, UE_FIVE = 5, UE_SIZE = 6, UE_SEVEN = 7, UE_EIGHT = 8, UE_NINE = 9,
};

enum class ScopedEnumTest {
  SE_ZERO = 0, SE_ONE, SE_TWO, SE_THREE, SE_FOUR, SE_FIVE, SE_SIZE, SE_SEVEN, SE_EIGHT, SE_NINE,
};

enum class TypedEnumTest : uint8_t {
  TE_ZERO = 0, TE_ONE, TE_TWO, TE_THREE, TE_FOUR, TE_FIVE, TE_SIZE, TE_SEVEN, TE_EIGHT, TE_NINE,
};

// Generate random enums.
inline UnscopedEnumTest getRandomValueForType(UnscopedEnumTest, int limit) {
    return static_cast<UnscopedEnumTest>(getRandomValue<unsigned>(std::min(10, limit)));
}

inline ScopedEnumTest getRandomValueForType(ScopedEnumTest, int limit) {
    return static_cast<ScopedEnumTest>(getRandomValue<unsigned>(std::min(10, limit)));
}

inline TypedEnumTest getRandomValueForType(TypedEnumTest, int limit) {
    return static_cast<TypedEnumTest>(getRandomValue<unsigned>(std::min(10, limit)));
}

/// Printable enums - boost test requires these to be in the std
namespace std {
  inline std::ostream &operator<<(std::ostream &stream, UnscopedEnumTest v) {  stream << "UnscopedEnum(" << int(v) << ")"; return stream;  }
  inline std::ostream &operator<<(std::ostream &stream, ScopedEnumTest v) {  stream << "ScopedEnum(" << int(v) << ")"; return stream;  }
  inline std::ostream &operator<<(std::ostream &stream, TypedEnumTest v) {  stream << "TypedEnum(" << int(v) << ")"; return stream;  }
}