//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <iostream>
#include <algorithm>

#include "TypesToTest.hpp"
#include "SimdTestUtilities.hpp"

using namespace xvec::simd;

enum class OperatorType {
    CONVERT_TO_UDT, CONVERT_FROM_UDT,
    ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULO,
    EQUAL, NOT_EQUAL, LESS, GREATER, LESS_EQUAL, GREATER_EQUAL,
    BIT_AND, BIT_OR, BIT_XOR, LEFT_SHIFT, RIGHT_SHIFT,
    LOGICAL_AND, LOGICAL_OR,
    UNARY_PLUS, UNARY_MINUS, UNARY_NOT, BIT_NOT,
    PRE_INCREMENT, POST_INCREMENT, PRE_DECREMENT, POST_DECREMENT,
    CUSTOM_CONVERT_TO_UDT, CUSTOM_CONVERT_FROM_UDT, CUSTOM_UNARY, CUSTOM_BINARY
};

inline const char* operatorName(OperatorType op) {
    switch(op) {
        case OperatorType::ADD: return "+";
        case OperatorType::SUBTRACT: return "-";
        case OperatorType::MULTIPLY: return "*";
        case OperatorType::DIVIDE: return "/";
        case OperatorType::MODULO: return "%";
        case OperatorType::EQUAL: return "==";
        case OperatorType::NOT_EQUAL: return "!=";
        case OperatorType::LESS: return "<";
        case OperatorType::GREATER: return ">";
        case OperatorType::LESS_EQUAL: return "<=";
        case OperatorType::GREATER_EQUAL: return ">=";
        case OperatorType::BIT_AND: return "&";
        case OperatorType::BIT_OR: return "|";
        case OperatorType::BIT_XOR: return "^";
        case OperatorType::LEFT_SHIFT: return "<<";
        case OperatorType::RIGHT_SHIFT: return ">>";
        case OperatorType::LOGICAL_AND: return "&&";
        case OperatorType::LOGICAL_OR: return "||";
        case OperatorType::UNARY_PLUS: return "+x";
        case OperatorType::UNARY_MINUS: return "-x";
        case OperatorType::UNARY_NOT: return "!x";
        case OperatorType::BIT_NOT: return "~x";
        case OperatorType::PRE_INCREMENT: return "++x";
        case OperatorType::POST_INCREMENT: return "x++";
        case OperatorType::PRE_DECREMENT: return "--x";
        case OperatorType::POST_DECREMENT: return "x--";
        case OperatorType::CUSTOM_UNARY: return "custom_unary";
        case OperatorType::CUSTOM_BINARY: return "custom_binary";
        case OperatorType::CONVERT_TO_UDT: return "convert_to_udt";
        case OperatorType::CONVERT_FROM_UDT: return "convert_from_udt";
        case OperatorType::CUSTOM_CONVERT_TO_UDT: return "custom_convert_to_udt";
        case OperatorType::CUSTOM_CONVERT_FROM_UDT: return "custom_convert_from_udt";
        default: return "UNKNOWN";
    }
}

struct OperatorCall {
    OperatorType op;
    int lhs;           // Left operand (or the only operand for unary)
    int rhs;           // Right operand (unused for unary)
    bool is_unary;

    OperatorCall(OperatorType op, int lhs) : op(op), lhs(lhs), rhs(0), is_unary(true) {}

    OperatorCall(OperatorType op, int lhs, int rhs) : op(op), lhs(lhs), rhs(rhs), is_unary(false) {}

    bool operator==(const OperatorCall& other) const {
        return op == other.op && lhs == other.lhs &&
               rhs == other.rhs && is_unary == other.is_unary;
    }

    friend std::ostream& operator<<(std::ostream& os, const OperatorCall& call) {
        os << operatorName(call.op) << "(" << call.lhs;
        if (!call.is_unary) os << ", " << call.rhs;
        os << ")";
        return os;
    }
};

class TestUDT {
public:
    int value;

    // Single vector tracking all operator calls in order
    inline static std::vector<OperatorCall> calls;

    TestUDT(int v = 0) : value(v) {}

    TestUDT(float x) {
        calls.push_back(OperatorCall(OperatorType::CONVERT_TO_UDT, x));
        value = x;
    }

    operator float() const {
        calls.push_back(OperatorCall(OperatorType::CONVERT_FROM_UDT, value));
        return value;
    }

    static void reset() {
        calls.clear();
    }

    static void printCalls() {
        std::cout << "Operator calls (" << calls.size() << "):\n";
        for (size_t i = 0; i < calls.size(); ++i) {
            std::cout << "  " << i << ": " << calls[i] << "\n";
        }
    }

    static size_t countCalls(OperatorType op) {
        return std::count_if(calls.begin(), calls.end(),
            [op](const OperatorCall& call) { return call.op == op; });
    }

    // ===== ARITHMETIC BINARY OPERATORS =====

    TestUDT operator+(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::ADD, value, other.value));
        return TestUDT(value + other.value);
    }

    TestUDT operator-(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::SUBTRACT, value, other.value));
        return TestUDT(value - other.value);
    }

    TestUDT operator*(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::MULTIPLY, value, other.value));
        return TestUDT(value * other.value);
    }

    TestUDT operator/(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::DIVIDE, value, other.value));
        return TestUDT(value / other.value);
    }

    TestUDT operator%(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::MODULO, value, other.value));
        return TestUDT(value % other.value);
    }

    // ===== COMPARISON OPERATORS =====

    bool operator==(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::EQUAL, value, other.value));
        return value == other.value;
    }

    bool operator!=(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::NOT_EQUAL, value, other.value));
        return value != other.value;
    }

    bool operator<(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::LESS, value, other.value));
        return value < other.value;
    }

    bool operator>(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::GREATER, value, other.value));
        return value > other.value;
    }

    bool operator<=(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::LESS_EQUAL, value, other.value));
        return value <= other.value;
    }

    bool operator>=(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::GREATER_EQUAL, value, other.value));
        return value >= other.value;
    }

    // ===== BITWISE BINARY OPERATORS =====

    TestUDT operator&(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::BIT_AND, value, other.value));
        return TestUDT(value & other.value);
    }

    TestUDT operator|(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::BIT_OR, value, other.value));
        return TestUDT(value | other.value);
    }

    TestUDT operator^(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::BIT_XOR, value, other.value));
        return TestUDT(value ^ other.value);
    }

    TestUDT operator<<(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::LEFT_SHIFT, value, other.value));
        return TestUDT(value << other.value);
    }

    TestUDT operator>>(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::RIGHT_SHIFT, value, other.value));
        return TestUDT(value >> other.value);
    }

    // ===== LOGICAL OPERATORS =====

    bool operator&&(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::LOGICAL_AND, value, other.value));
        return value && other.value;
    }

    bool operator||(const TestUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::LOGICAL_OR, value, other.value));
        return value || other.value;
    }

    // ===== UNARY OPERATORS =====

    TestUDT operator+() const {
        calls.push_back(OperatorCall(OperatorType::UNARY_PLUS, value));
        return TestUDT(+value);
    }

    TestUDT operator-() const {
        calls.push_back(OperatorCall(OperatorType::UNARY_MINUS, value));
        return TestUDT(-value);
    }

    bool operator!() const {
        calls.push_back(OperatorCall(OperatorType::UNARY_NOT, value));
        return !value;
    }

    TestUDT operator~() const {
        calls.push_back(OperatorCall(OperatorType::BIT_NOT, value));
        return TestUDT(~value);
    }

    // Pre-increment
    TestUDT& operator++() {
        calls.push_back(OperatorCall(OperatorType::PRE_INCREMENT, value));
        ++value;
        return *this;
    }

    // Post-increment
    TestUDT operator++(int) {
        calls.push_back(OperatorCall(OperatorType::POST_INCREMENT, value));
        TestUDT temp(value);
        ++value;
        return temp;
    }

    // Pre-decrement
    TestUDT& operator--() {
        calls.push_back(OperatorCall(OperatorType::PRE_DECREMENT, value));
        --value;
        return *this;
    }

    // Post-decrement
    TestUDT operator--(int) {
        calls.push_back(OperatorCall(OperatorType::POST_DECREMENT, value));
        TestUDT temp(value);
        --value;
        return temp;
    }
};

/// Custom UDT. This is used to check that if a customisation function is
/// called, it will be used in preference to the TestUDT::operator.
struct CustomisedUDT : public TestUDT {

    CustomisedUDT(int v = 0) : TestUDT(v) {}

    CustomisedUDT(float x) {
        calls.push_back(OperatorCall(OperatorType::CONVERT_TO_UDT, x));
        value = x;
    }

    operator float() const {
        calls.push_back(OperatorCall(OperatorType::CONVERT_FROM_UDT, value));
        return value;
    }

    CustomisedUDT operator-() const {
        calls.push_back(OperatorCall(OperatorType::UNARY_MINUS, value));
        return CustomisedUDT(-value);
    }

    CustomisedUDT operator+(const CustomisedUDT& other) const {
        calls.push_back(OperatorCall(OperatorType::ADD, value, other.value));
        return CustomisedUDT(value + other.value);
    }
};

struct SimdUdtTestFixture {

    const vec<TestUDT, 3> a = std::array<TestUDT, 3>{{TestUDT(0), TestUDT(1), TestUDT(2)}};
    const vec<TestUDT, 3> b = std::array<TestUDT, 3>{{TestUDT(3), TestUDT(4), TestUDT(5)}};

    SimdUdtTestFixture() { TestUDT::reset(); }
    ~SimdUdtTestFixture() { TestUDT::reset(); }

    // Helper to verify exact sequence of calls
    void verifyExactCalls(const std::vector<OperatorCall>& expected) {
        BOOST_REQUIRE_EQUAL(TestUDT::calls.size(), expected.size());

        for (size_t i = 0; i < expected.size(); ++i) {
            BOOST_TEST_CONTEXT("Call index " << i) {
                BOOST_CHECK(TestUDT::calls[i] == expected[i]);
                if (!(TestUDT::calls[i] == expected[i])) {
                    std::cout << "Expected: " << expected[i] << "\n";
                    std::cout << "Got:      " << TestUDT::calls[i] << "\n";
                }
            }
        }
    }
};

BOOST_FIXTURE_TEST_SUITE(ArrayOperatorTests, SimdUdtTestFixture)

BOOST_AUTO_TEST_CASE(test_convert_to_udt)
{
    vec<float, 3> cf = std::array<float, 3>{0, 1, 2};
    [[maybe_unused]] vec<TestUDT, 3> result = cf;

    verifyExactCalls({
        OperatorCall(OperatorType::CONVERT_TO_UDT, 0),
        OperatorCall(OperatorType::CONVERT_TO_UDT, 1),
        OperatorCall(OperatorType::CONVERT_TO_UDT, 2)
    });
}

BOOST_AUTO_TEST_CASE(test_convert_from_udt)
{
    [[maybe_unused]] vec<float, 3> cf = a;

    verifyExactCalls({
        OperatorCall(OperatorType::CONVERT_FROM_UDT, 0),
        OperatorCall(OperatorType::CONVERT_FROM_UDT, 1),
        OperatorCall(OperatorType::CONVERT_FROM_UDT, 2)
    });
}

// ===== ARITHMETIC BINARY OPERATORS =====

BOOST_AUTO_TEST_CASE(test_addition)
{
    [[maybe_unused]] auto result = a + b;

    verifyExactCalls({
        OperatorCall(OperatorType::ADD, 0, 3),
        OperatorCall(OperatorType::ADD, 1, 4),
        OperatorCall(OperatorType::ADD, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_subtraction)
{
    [[maybe_unused]] auto result = a - b;

    verifyExactCalls({
        OperatorCall(OperatorType::SUBTRACT, 0, 3),
        OperatorCall(OperatorType::SUBTRACT, 1, 4),
        OperatorCall(OperatorType::SUBTRACT, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_multiplication)
{
    [[maybe_unused]] auto result = a * b;

    verifyExactCalls({
        OperatorCall(OperatorType::MULTIPLY, 0, 3),
        OperatorCall(OperatorType::MULTIPLY, 1, 4),
        OperatorCall(OperatorType::MULTIPLY, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_division)
{
    [[maybe_unused]] auto result = a / b;

    verifyExactCalls({
        OperatorCall(OperatorType::DIVIDE, 0, 3),
        OperatorCall(OperatorType::DIVIDE, 1, 4),
        OperatorCall(OperatorType::DIVIDE, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_modulo)
{
    [[maybe_unused]] auto result = a % b;

    verifyExactCalls({
        OperatorCall(OperatorType::MODULO, 0, 3),
        OperatorCall(OperatorType::MODULO, 1, 4),
        OperatorCall(OperatorType::MODULO, 2, 5)
    });
}

// ===== COMPARISON OPERATORS =====

BOOST_AUTO_TEST_CASE(test_less_than)
{
    [[maybe_unused]] auto result = a < b;

    verifyExactCalls({
        OperatorCall(OperatorType::LESS, 0, 3),
        OperatorCall(OperatorType::LESS, 1, 4),
        OperatorCall(OperatorType::LESS, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_greater_than)
{
    [[maybe_unused]] auto result = a > b;

    verifyExactCalls({
        OperatorCall(OperatorType::GREATER, 0, 3),
        OperatorCall(OperatorType::GREATER, 1, 4),
        OperatorCall(OperatorType::GREATER, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_less_equal)
{
    [[maybe_unused]] auto result = a <= b;

    verifyExactCalls({
        OperatorCall(OperatorType::LESS_EQUAL, 0, 3),
        OperatorCall(OperatorType::LESS_EQUAL, 1, 4),
        OperatorCall(OperatorType::LESS_EQUAL, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_greater_equal)
{
    [[maybe_unused]] auto result = a >= b;

    verifyExactCalls({
        OperatorCall(OperatorType::GREATER_EQUAL, 0, 3),
        OperatorCall(OperatorType::GREATER_EQUAL, 1, 4),
        OperatorCall(OperatorType::GREATER_EQUAL, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_equality)
{
    [[maybe_unused]] auto result = a == b;

    verifyExactCalls({
        OperatorCall(OperatorType::EQUAL, 0, 3),
        OperatorCall(OperatorType::EQUAL, 1, 4),
        OperatorCall(OperatorType::EQUAL, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_inequality)
{
    [[maybe_unused]] auto result = a != b;

    verifyExactCalls({
        OperatorCall(OperatorType::NOT_EQUAL, 0, 3),
        OperatorCall(OperatorType::NOT_EQUAL, 1, 4),
        OperatorCall(OperatorType::NOT_EQUAL, 2, 5)
    });
}

// ===== BITWISE BINARY OPERATORS =====

BOOST_AUTO_TEST_CASE(test_bitwise_and)
{
    [[maybe_unused]] auto result = a & b;

    verifyExactCalls({
        OperatorCall(OperatorType::BIT_AND, 0, 3),
        OperatorCall(OperatorType::BIT_AND, 1, 4),
        OperatorCall(OperatorType::BIT_AND, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_bitwise_or)
{
    [[maybe_unused]] auto result = a | b;

    verifyExactCalls({
        OperatorCall(OperatorType::BIT_OR, 0, 3),
        OperatorCall(OperatorType::BIT_OR, 1, 4),
        OperatorCall(OperatorType::BIT_OR, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_bitwise_xor)
{
    [[maybe_unused]] auto result = a ^ b;

    verifyExactCalls({
        OperatorCall(OperatorType::BIT_XOR, 0, 3),
        OperatorCall(OperatorType::BIT_XOR, 1, 4),
        OperatorCall(OperatorType::BIT_XOR, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_left_shift)
{
    [[maybe_unused]] auto result = a << b;

    verifyExactCalls({
        OperatorCall(OperatorType::LEFT_SHIFT, 0, 3),
        OperatorCall(OperatorType::LEFT_SHIFT, 1, 4),
        OperatorCall(OperatorType::LEFT_SHIFT, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_right_shift)
{
    [[maybe_unused]] auto result = a >> b;

    verifyExactCalls({
        OperatorCall(OperatorType::RIGHT_SHIFT, 0, 3),
        OperatorCall(OperatorType::RIGHT_SHIFT, 1, 4),
        OperatorCall(OperatorType::RIGHT_SHIFT, 2, 5)
    });
}

// ===== UNARY OPERATORS =====

// BOOST_AUTO_TEST_CASE(test_unary_plus)
// {
//     [[maybe_unused]] auto result = +a;

//     verifyExactCalls({
//         OperatorCall(OperatorType::UNARY_PLUS, 0),
//         OperatorCall(OperatorType::UNARY_PLUS, 1),
//         OperatorCall(OperatorType::UNARY_PLUS, 2)
//     });
// }

BOOST_AUTO_TEST_CASE(test_unary_minus)
{
    [[maybe_unused]] auto result = -a;

    verifyExactCalls({
        OperatorCall(OperatorType::UNARY_MINUS, 0),
        OperatorCall(OperatorType::UNARY_MINUS, 1),
        OperatorCall(OperatorType::UNARY_MINUS, 2)
    });
}

BOOST_AUTO_TEST_CASE(test_logical_not)
{
    [[maybe_unused]] auto result = !a;

    verifyExactCalls({
        OperatorCall(OperatorType::UNARY_NOT, 0),
        OperatorCall(OperatorType::UNARY_NOT, 1),
        OperatorCall(OperatorType::UNARY_NOT, 2)
    });
}

BOOST_AUTO_TEST_CASE(test_bitwise_not)
{
    [[maybe_unused]] auto result = ~a;

    verifyExactCalls({
        OperatorCall(OperatorType::BIT_NOT, 0),
        OperatorCall(OperatorType::BIT_NOT, 1),
        OperatorCall(OperatorType::BIT_NOT, 2)
    });
}

// BOOST_AUTO_TEST_CASE(test_pre_increment)
// {
//     // Need a mutable copy since a is const
//     [[maybe_unused]] auto a_copy = a;

//     ++a_copy;

//     verifyExactCalls({
//         OperatorCall(OperatorType::PRE_INCREMENT, 0),
//         OperatorCall(OperatorType::PRE_INCREMENT, 1),
//         OperatorCall(OperatorType::PRE_INCREMENT, 2)
//     });

//     // Verify values were actually incremented
//     BOOST_CHECK_EQUAL(a_copy[0].value, 1);
//     BOOST_CHECK_EQUAL(a_copy[1].value, 2);
//     BOOST_CHECK_EQUAL(a_copy[2].value, 3);
// }

// BOOST_AUTO_TEST_CASE(test_post_increment)
// {
//     // Need a mutable copy since a is const
//     [[maybe_unused]] auto a_copy = a;

//     [[maybe_unused]] auto old = a_copy++;

//     verifyExactCalls({
//         OperatorCall(OperatorType::POST_INCREMENT, 0),
//         OperatorCall(OperatorType::POST_INCREMENT, 1),
//         OperatorCall(OperatorType::POST_INCREMENT, 2)
//     });

//     // Verify values were incremented
//     BOOST_CHECK_EQUAL(a_copy[0].value, 1);
//     BOOST_CHECK_EQUAL(a_copy[1].value, 2);
//     BOOST_CHECK_EQUAL(a_copy[2].value, 3);
// }

// BOOST_AUTO_TEST_CASE(test_pre_decrement)
// {
//     // Need a mutable copy since a is const
//     [[maybe_unused]] auto a_copy = a;

//     --a_copy;

//     verifyExactCalls({
//         OperatorCall(OperatorType::PRE_DECREMENT, 0),
//         OperatorCall(OperatorType::PRE_DECREMENT, 1),
//         OperatorCall(OperatorType::PRE_DECREMENT, 2)
//     });

//     // Verify values were actually decremented
//     BOOST_CHECK_EQUAL(a_copy[0].value, -1);
//     BOOST_CHECK_EQUAL(a_copy[1].value, 0);
//     BOOST_CHECK_EQUAL(a_copy[2].value, 1);
// }

// BOOST_AUTO_TEST_CASE(test_post_decrement)
// {
//     // Need a mutable copy since a is const
//     [[maybe_unused]] auto a_copy = a;

//     [[maybe_unused]] auto old = a_copy--;

//     verifyExactCalls({
//         OperatorCall(OperatorType::POST_DECREMENT, 0),
//         OperatorCall(OperatorType::POST_DECREMENT, 1),
//         OperatorCall(OperatorType::POST_DECREMENT, 2)
//     });

//     // Verify values were decremented
//     BOOST_CHECK_EQUAL(a_copy[0].value, -1);
//     BOOST_CHECK_EQUAL(a_copy[1].value, 0);
//     BOOST_CHECK_EQUAL(a_copy[2].value, 1);
// }

BOOST_AUTO_TEST_CASE(test_shift_operators_comprehensive)
{
    // Left shift: should call operator<< on each element
    [[maybe_unused]] auto left_shifted = a << b;

    verifyExactCalls({
        OperatorCall(OperatorType::LEFT_SHIFT, 0, 3),
        OperatorCall(OperatorType::LEFT_SHIFT, 1, 4),
        OperatorCall(OperatorType::LEFT_SHIFT, 2, 5)
    });

    TestUDT::reset();

    // Right shift: should call operator>> on each element
    [[maybe_unused]] auto right_shifted = a >> b;

    verifyExactCalls({
        OperatorCall(OperatorType::RIGHT_SHIFT, 0, 3),
        OperatorCall(OperatorType::RIGHT_SHIFT, 1, 4),
        OperatorCall(OperatorType::RIGHT_SHIFT, 2, 5)
    });
}

BOOST_AUTO_TEST_CASE(test_bitwise_operations_comprehensive)
{
    // Test AND
    [[maybe_unused]] auto and_result = a & b;
    BOOST_CHECK_EQUAL(TestUDT::countCalls(OperatorType::BIT_AND), 3);

    TestUDT::reset();

    // Test OR
    [[maybe_unused]] auto or_result = a | b;
    BOOST_CHECK_EQUAL(TestUDT::countCalls(OperatorType::BIT_OR), 3);

    TestUDT::reset();

    // Test XOR
    [[maybe_unused]] auto xor_result = a ^ b;
    BOOST_CHECK_EQUAL(TestUDT::countCalls(OperatorType::BIT_XOR), 3);

    TestUDT::reset();

    // Test NOT (unary)
    [[maybe_unused]] auto not_result = ~a;
    BOOST_CHECK_EQUAL(TestUDT::countCalls(OperatorType::BIT_NOT), 3);
}

BOOST_AUTO_TEST_CASE(test_min)
{
    [[maybe_unused]] auto result = min(a, b);
    // The min function may change the order of the operands, so we just check
    // that the less-than operator was called 3 times, without assuming the order.
    BOOST_CHECK_EQUAL(TestUDT::countCalls(OperatorType::LESS), 3);
}

BOOST_AUTO_TEST_CASE(test_max)
{
    [[maybe_unused]] auto result = max(a, b);
    // The max function may change the order of the operands, so we just check
    // that the less-than operator was called 3 times, without assuming the order.
    BOOST_CHECK_EQUAL(TestUDT::countCalls(OperatorType::LESS), 3);
}

BOOST_AUTO_TEST_SUITE_END()

// UDT Customisation Points

template<vec_of<float> _From>
auto simd_convert(const _From&, convert_to_t<CustomisedUDT>)
{
    CustomisedUDT::calls.push_back(OperatorCall(OperatorType::CUSTOM_CONVERT_TO_UDT, 0));
    return rebind_t<CustomisedUDT, _From>();
}

template<vec_of<CustomisedUDT> _From>
auto simd_convert(const _From&, convert_to_t<float>)
{
    CustomisedUDT::calls.push_back(OperatorCall(OperatorType::CUSTOM_CONVERT_FROM_UDT, 0));
    return rebind_t<float, _From>();
}

template<vec_of<CustomisedUDT> _Vp> _Vp simd_operator(_Vp, std::negate<>) {
    CustomisedUDT::calls.push_back(OperatorCall(OperatorType::CUSTOM_UNARY, 0));
    return _Vp();
}

template<vec_of<CustomisedUDT> _Vp>
_Vp simd_operator(_Vp, _Vp, std::plus<>) {
    CustomisedUDT::calls.push_back(OperatorCall(OperatorType::CUSTOM_BINARY, 0, 0));
    return _Vp();
}

struct CustomisedUdtTestFixture {
    const vec<CustomisedUDT, 3> a = std::array<CustomisedUDT, 3>{{CustomisedUDT(10), CustomisedUDT(20), CustomisedUDT(30)}};
    const vec<CustomisedUDT, 3> b = std::array<CustomisedUDT, 3>{{CustomisedUDT(1), CustomisedUDT(2), CustomisedUDT(3)}};

    CustomisedUdtTestFixture() { CustomisedUDT::reset(); }
    ~CustomisedUdtTestFixture() { CustomisedUDT::reset(); }
};

BOOST_FIXTURE_TEST_SUITE(CustomisedUdtTests, CustomisedUdtTestFixture)

BOOST_AUTO_TEST_CASE(test_convert_to_udt)
{
    vec<float, 3> cf = std::array<float, 3>{0, 1, 2};
    [[maybe_unused]] vec<CustomisedUDT, 3> result = cf;

    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::CUSTOM_CONVERT_TO_UDT), 1);
    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::CONVERT_TO_UDT), 0);
}

BOOST_AUTO_TEST_CASE(test_convert_from_udt)
{
    [[maybe_unused]] vec<float, 3> cf = a;

    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::CUSTOM_CONVERT_FROM_UDT), 1);
    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::CONVERT_FROM_UDT), 0);
}

BOOST_AUTO_TEST_CASE(test_customised_unary_minus)
{
    [[maybe_unused]] auto result = -a;

    // Should call custom unary_minus, not CustomisedUDT::operator-
    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::CUSTOM_UNARY), 1);
    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::UNARY_MINUS), 0);
}

BOOST_AUTO_TEST_CASE(test_customised_binary_plus)
{
    [[maybe_unused]] auto result = a + b;

    // Should call custom binary_plus, not CustomisedUDT::operator+
    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::CUSTOM_BINARY), 1);
    BOOST_CHECK_EQUAL(CustomisedUDT::countCalls(OperatorType::ADD), 0);
}

BOOST_AUTO_TEST_SUITE_END()