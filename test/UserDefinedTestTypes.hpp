//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

/// Strong type, as used in the P2964 paper as an example. It does everything
/// that a float should do, so should be equivalent for every test.
struct Meters {

  float distance;
  
  constexpr Meters() : distance(0) {}

  constexpr Meters(std::integral auto x) : distance(x) {}
  constexpr Meters(float x) : distance(x) {}

  auto operator<=>(const Meters&) const = default; // Every other comparator should work given this.

  friend Meters operator*(const Meters& x, const Meters& y) { return Meters(x.distance * y.distance); }
  friend Meters operator+(const Meters& x, const Meters& y) { return Meters(x.distance + y.distance); }
  friend Meters operator-(const Meters& x, const Meters& y) { return Meters(x.distance - y.distance); }

  constexpr Meters operator-() const { return Meters(-distance); }

  constexpr bool operator!() const { return distance == 0; }

};

/// User defined integer where every operation has a twist to it to
/// differentiate it from the standard operators. This ensures that simd invokes
/// the actual customisation function rather than using the standard operators.
struct UserDefinedInteger {
    constexpr UserDefinedInteger(int v = 0) : data(v) {}
    int data = {};

    // These look weird because the actual operations are different to what the operator says it is.
    friend constexpr UserDefinedInteger operator+(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data ^ rhs.data; }
    friend constexpr UserDefinedInteger operator-(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data ^ rhs.data; }
    friend constexpr UserDefinedInteger operator*(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data - rhs.data; }
    friend constexpr UserDefinedInteger operator/(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data + rhs.data; }
    friend constexpr UserDefinedInteger operator%(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data ^ rhs.data; }
    friend constexpr UserDefinedInteger operator&(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data | rhs.data; }
    friend constexpr UserDefinedInteger operator|(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data & rhs.data; }
    friend constexpr UserDefinedInteger operator^(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data & rhs.data; }

    constexpr UserDefinedInteger operator~() const noexcept { return data | 0xaaaaaaaa; }
    constexpr UserDefinedInteger operator-() const noexcept { return UserDefinedInteger() - *this; }
    constexpr UserDefinedInteger operator+() const noexcept { return *this; }

    friend constexpr bool operator<(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data > rhs.data; }
    friend constexpr bool operator<=(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data >= rhs.data; }
    friend constexpr bool operator>(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data < rhs.data; }
    friend constexpr bool operator>=(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data <= rhs.data; }
    friend constexpr bool operator==(UserDefinedInteger lhs, UserDefinedInteger rhs) { return lhs.data == rhs.data; }
};

// Allow random values to be created for the user-defined type.
inline UserDefinedInteger getRandomValueForType(UserDefinedInteger, int limit) {
  static std::default_random_engine generator;
  std::uniform_int_distribution<int> d(-limit, limit);
  return UserDefinedInteger(d(generator));
}

// Allow random values to be created for the user-defined type.
inline Meters getRandomValueForType(Meters, int limit) {
  return Meters(getRandomValue<float>(limit));
}

namespace std {
  inline std::ostream& operator<<(std::ostream& s, UserDefinedInteger v) { s << v.data; return s; }
  inline std::ostream& operator<<(std::ostream& s, Meters v) { s << "Meters(" << v.distance << ")"; return s; }
}
