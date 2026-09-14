##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

# Standalone unit-test configuration for xvec.
#
# ── Compiler selection ────────────────────────────────────────────────────────
# Supply the compiler at configure time:
#   cmake -S test -B build -DCMAKE_CXX_COMPILER=g++-13
#   cmake -S test -B build -DCMAKE_CXX_COMPILER=clang++-17
#   cmake -S test -B build -DCMAKE_CXX_COMPILER=icpx
#
# Recognized compiler families:
#   GNU, Clang, IntelLLVM
#
# ── Processor matrix ──────────────────────────────────────────────────────────
# Defined below via xvec_processor().
# For every *Test*.cpp source and selected processor, one executable and one
# CTest test are created. Tests run through SDE emulation:
#   sde64 -<sde_cpu> -- <test-executable>
#
# FAST tests use a reduced processor set, configurable via:
#   -DXVEC_FAST_PROCESSORS="generic;avx2;avxglc"
#
# ── Standard workflow ─────────────────────────────────────────────────────────
#   cmake -S test -B build -DCMAKE_CXX_COMPILER=g++
#   cmake --build build --parallel
#   ctest --test-dir build --output-on-failure
#
# ── Filtering examples ────────────────────────────────────────────────────────
#   ctest --test-dir build -L avx512
#   ctest --test-dir build -L TestSimdMath
#   ctest --test-dir build -R TestSimdMath
#   ctest --test-dir build -L avx512 --parallel

cmake_minimum_required(VERSION 3.20)

project(XVEC VERSION 0.1 LANGUAGES CXX)

# All xvec tests and compiler checks use C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

get_filename_component(
  XVEC_ROOT_DIR
  "${CMAKE_CURRENT_LIST_DIR}/.."
  ABSOLUTE
)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|IntelLLVM)$")
  message(
    FATAL_ERROR
    "Unsupported C++ compiler family '${CMAKE_CXX_COMPILER_ID}'. "
    "Supported families are GNU, Clang, and IntelLLVM."
  )
endif()

# ── Intel SDE ─────────────────────────────────────────────────────────────────
# XVEC_SDE_EXECUTABLE is cached by find_program(), so it can also be supplied:
#   cmake ... -DXVEC_SDE_EXECUTABLE=/path/to/sde64
find_program(
  XVEC_SDE_EXECUTABLE
  NAMES sde64
  DOC "Path to the Intel(R) Software Development Emulator (Intel(R) SDE) executable"
)

if(NOT XVEC_SDE_EXECUTABLE)
  message(
    FATAL_ERROR
    "Intel SDE executable 'sde64' was not found. "
    "Add it to PATH or configure with "
    "-DXVEC_SDE_EXECUTABLE=/path/to/sde64."
  )
endif()

enable_testing()

# ── Test configuration ────────────────────────────────────────────────────────
# Controls how many type/size combinations each test exercises:
#   NATIVE: simd<T> for all T only
#   FAST:   all types, a few sizes, and a reduced processor set
#   FULL:   all types, many sizes, and all registered processors
set(
  XVEC_TEST_TYPE
  "FULL"
  CACHE STRING
  "Test scope: NATIVE, FAST, or FULL"
)

set_property(
  CACHE XVEC_TEST_TYPE
  PROPERTY STRINGS NATIVE FAST FULL
)

# Processors included in FAST test configurations. This is a semicolon-
# separated cache variable, so it can be overridden at configure time:
#   cmake ... -DXVEC_TEST_TYPE=FAST \
#             -DXVEC_FAST_PROCESSORS="generic;avx2;avxglc"
set(
  XVEC_FAST_PROCESSORS
  "generic;avx2;avxglc"
  CACHE STRING
  "Processors included when XVEC_TEST_TYPE is FAST"
)

string(TOUPPER "${XVEC_TEST_TYPE}" XVEC_TEST_TYPE)

if(NOT XVEC_TEST_TYPE MATCHES "^(NATIVE|FAST|FULL)$")
  message(
    FATAL_ERROR
    "Invalid XVEC_TEST_TYPE='${XVEC_TEST_TYPE}'. "
    "Expected NATIVE, FAST, or FULL."
  )
endif()

# Building above -O0 can make compile times prohibitive for FULL tests.
set(
  XVEC_TEST_OLEVEL
  "-O0"
  CACHE STRING
  "Compiler optimization level for unit tests"
)

set(
  XVEC_TEST_TIMEOUT
  "300"
  CACHE STRING
  "Timeout in seconds for each SDE-based unit test"
)

set(
  XVEC_NUM_PARALLEL_TEST_JOBS
  "1"
  CACHE STRING
  "Maximum number of tests run concurrently by the check target"
)

if(NOT XVEC_TEST_TIMEOUT MATCHES "^[1-9][0-9]*$")
  message(
    FATAL_ERROR
    "XVEC_TEST_TIMEOUT must be a positive integer, got "
    "'${XVEC_TEST_TIMEOUT}'."
  )
endif()

if(NOT XVEC_NUM_PARALLEL_TEST_JOBS MATCHES "^[1-9][0-9]*$")
  message(
    FATAL_ERROR
    "XVEC_NUM_PARALLEL_TEST_JOBS must be a positive integer, got "
    "'${XVEC_NUM_PARALLEL_TEST_JOBS}'."
  )
endif()

# ── Test-scope definitions ────────────────────────────────────────────────────
set(_xvec_test_definitions)

if(XVEC_TEST_TYPE STREQUAL "NATIVE")
  list(APPEND _xvec_test_definitions _XVEC_TEST_NATIVE)
  message(STATUS "xvec tests: NATIVE scope (simd<T> only)")
elseif(XVEC_TEST_TYPE STREQUAL "FAST")
  list(APPEND _xvec_test_definitions _XVEC_TEST_FAST)
  message(
    STATUS
    "xvec tests: FAST scope (all types, few sizes, reduced processors)"
  )
else()
  message(
    STATUS
    "xvec tests: FULL scope "
    "(all types, many sizes; use NATIVE or FAST to reduce build time)"
  )
endif()

# ── Constexpr vector-extension detection ─────────────────────────────────────
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# The processor flag applies only to this compiler capability check.
# C++20 is inherited from the project-wide CMAKE_CXX_STANDARD setting.
cmake_push_check_state(RESET)

set(CMAKE_REQUIRED_FLAGS "-march=broadwell")

check_cxx_source_compiles(
  [=[
    using vec4i = int __attribute__((vector_size(16)));

    constexpr int extract_element()
    {
      vec4i value{1, 2, 3, 4};
      return value[3];
    }

    static_assert(extract_element() == 4);

    int main()
    {
      return 0;
    }
  ]=]
  XVEC_HAS_CONSTEXPR_VECTOR_ACCESS
)

cmake_pop_check_state()

if(XVEC_HAS_CONSTEXPR_VECTOR_ACCESS)
  list(APPEND _xvec_test_definitions _XVEC_HAS_CONSTEXPR)
  message(STATUS "xvec: constexpr vector access is supported")
else()
  message(STATUS "xvec: constexpr vector access is not supported")
endif()

# ── Common test configuration target ─────────────────────────────────────────
# All test executables and TestMain object libraries link against this target.
add_library(xvec_test_common INTERFACE)

# Boost.Test is expected to be supplied by the toolchain or system; add its
# include directory to CMAKE_CXX_FLAGS or CMAKE_PREFIX_PATH if it lives
# somewhere the compiler does not search by default.
target_include_directories(
  xvec_test_common
  INTERFACE
    "${XVEC_ROOT_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(
  xvec_test_common
  INTERFACE
    -Wall
    -Wextra
    -Wno-tautological-constant-compare
    -Wno-deprecated-declarations
    -Wno-psabi
    -Wno-sign-compare
    "${XVEC_TEST_OLEVEL}"
    -fno-fast-math
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(IntelLLVM|Clang)$")
  target_compile_options(
    xvec_test_common
    INTERFACE
      -Wno-unknown-warning-option
      -Wno-deprecated-copy
      -Wno-deprecated-builtins
      -Wno-unused-command-line-argument
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(
      xvec_test_common
      INTERFACE
        -Wno-deprecated-type
    )
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    xvec_test_common
    INTERFACE
      -Wno-unknown-warning
      -Wno-maybe-uninitialized
      -Wno-unknown-pragmas
      -flax-vector-conversions
  )
endif()

target_compile_definitions(
  xvec_test_common
  INTERFACE
    ${_xvec_test_definitions}
)

# ── Processor registration helper ─────────────────────────────────────────────
# xvec_processor(
#   <proc_id>
#   SDE_CPU <cpu>
#   COMPILE_FLAGS <flag> [<flag>...]
#   [COMPILE_DEFINITIONS <definition> [<definition>...]]
# )
#
# Each processor is represented by an interface target. This keeps processor
# options attached to targets instead of passing variable names through nested
# loops.
function(xvec_processor proc_id)
  cmake_parse_arguments(
    PARSE_ARGV 1
    ARG
    ""
    "SDE_CPU"
    "COMPILE_FLAGS;COMPILE_DEFINITIONS"
  )

  if(ARG_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "xvec_processor(${proc_id}) received unknown arguments: "
      "${ARG_UNPARSED_ARGUMENTS}"
    )
  endif()

  if(NOT ARG_SDE_CPU)
    message(
      FATAL_ERROR
      "xvec_processor(${proc_id}) requires SDE_CPU."
    )
  endif()

  if(NOT ARG_COMPILE_FLAGS)
    message(
      FATAL_ERROR
      "xvec_processor(${proc_id}) requires COMPILE_FLAGS."
    )
  endif()

  set(_processor_target "xvec_processor_${proc_id}")

  if(TARGET "${_processor_target}")
    message(
      FATAL_ERROR
      "Processor '${proc_id}' has already been registered."
    )
  endif()

  add_library("${_processor_target}" INTERFACE)

  target_compile_options(
    "${_processor_target}"
    INTERFACE
      ${ARG_COMPILE_FLAGS}
  )

  target_compile_definitions(
    "${_processor_target}"
    INTERFACE
      "_XVEC_PROC_${proc_id}"
      ${ARG_COMPILE_DEFINITIONS}
  )

  set_property(
    GLOBAL
    APPEND
    PROPERTY XVEC_PROCESSOR_IDS
    "${proc_id}"
  )

  set_property(
    GLOBAL
    PROPERTY "XVEC_PROCESSOR_SDE_${proc_id}"
    "${ARG_SDE_CPU}"
  )
endfunction()

# ── Processor matrix ──────────────────────────────────────────────────────────
xvec_processor(
  sse
  COMPILE_FLAGS -march=goldmont
  SDE_CPU glm
)

xvec_processor(
  avx
  COMPILE_FLAGS -march=core-avx-i
  SDE_CPU ivb
)

xvec_processor(
  avx2
  COMPILE_FLAGS -march=core-avx2
  SDE_CPU bdw
)

xvec_processor(
  avx512
  COMPILE_FLAGS -march=skylake-avx512
  SDE_CPU skx
)

xvec_processor(
  avxsnc
  COMPILE_FLAGS -march=icelake-server
  SDE_CPU icx
)

xvec_processor(
  avxglc
  COMPILE_FLAGS -march=sapphirerapids
  SDE_CPU spr
)

xvec_processor(
  generic
  COMPILE_FLAGS -march=skylake-avx512
  COMPILE_DEFINITIONS _XVEC_FORCE_SCALAR
  SDE_CPU spr
)

xvec_processor(
  generic_compact
  COMPILE_FLAGS -march=skylake-avx512
  COMPILE_DEFINITIONS _XVEC_FORCE_SCALAR _XVEC_GENERIC_COMPACT
  SDE_CPU spr
)

get_property(
  _xvec_processor_ids
  GLOBAL
  PROPERTY XVEC_PROCESSOR_IDS
)

if(NOT _xvec_processor_ids)
  message(FATAL_ERROR "No xvec processors were registered.")
endif()

# FAST configurations build and run only the selected processor subset.
if(XVEC_TEST_TYPE STREQUAL "FAST")
  if(NOT XVEC_FAST_PROCESSORS)
    message(
      FATAL_ERROR
      "XVEC_FAST_PROCESSORS must contain at least one processor."
    )
  endif()

  set(_xvec_fast_processor_ids ${XVEC_FAST_PROCESSORS})
  list(REMOVE_DUPLICATES _xvec_fast_processor_ids)

  foreach(_proc_id IN LISTS _xvec_fast_processor_ids)
    if(NOT _proc_id IN_LIST _xvec_processor_ids)
      message(
        FATAL_ERROR
        "Unknown processor '${_proc_id}' in XVEC_FAST_PROCESSORS. "
        "Registered processors: ${_xvec_processor_ids}"
      )
    endif()
  endforeach()

  set(_xvec_processor_ids ${_xvec_fast_processor_ids})

  message(
    STATUS
    "xvec tests: FAST processors=${_xvec_processor_ids}"
  )
endif()

# ── Discover test sources ─────────────────────────────────────────────────────
# TestMain.cpp supplies main() through Boost.Test. It is compiled once for each
# selected processor rather than once for every processor/test-source pair.
file(
  GLOB
  _xvec_test_sources
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*Test*.cpp"
)

list(
  FILTER
  _xvec_test_sources
  EXCLUDE
  REGEX "TestMain\\.cpp$"
)

if(NOT _xvec_test_sources)
  message(
    WARNING
    "No test sources matching '*Test*.cpp' were found in "
    "${CMAKE_CURRENT_SOURCE_DIR}."
  )
endif()

# ── Build TestMain once per processor ─────────────────────────────────────────
foreach(_proc_id IN LISTS _xvec_processor_ids)
  set(_main_target "xvec_test_main_${_proc_id}")

  add_library(
    "${_main_target}"
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/TestMain.cpp"
  )

  target_link_libraries(
    "${_main_target}"
    PRIVATE
      xvec_test_common
      "xvec_processor_${_proc_id}"
  )
endforeach()

# ── Build and register tests ──────────────────────────────────────────────────
# Target name: <proc_id>-<TestName>
# CTest name:  <proc_id>/<TestName>
# Labels:      <proc_id> and <TestName>
set(_xvec_all_test_targets)

foreach(_src IN LISTS _xvec_test_sources)
  get_filename_component(_test_name "${_src}" NAME_WE)

  foreach(_proc_id IN LISTS _xvec_processor_ids)
    set(_target "${_proc_id}-${_test_name}")
    set(_main_target "xvec_test_main_${_proc_id}")

    get_property(
      _sde_cpu
      GLOBAL
      PROPERTY "XVEC_PROCESSOR_SDE_${_proc_id}"
    )

    if(NOT _sde_cpu)
      message(
        FATAL_ERROR
        "No SDE CPU was registered for processor '${_proc_id}'."
      )
    endif()

    add_executable(
      "${_target}"
      "${_src}"
      $<TARGET_OBJECTS:${_main_target}>
    )

    target_link_libraries(
      "${_target}"
      PRIVATE
        xvec_test_common
        "xvec_processor_${_proc_id}"
    )

    add_test(
      NAME "${_proc_id}/${_test_name}"
      COMMAND
        "${XVEC_SDE_EXECUTABLE}"
        "-${_sde_cpu}"
        --
        "$<TARGET_FILE:${_target}>"
    )

    set_tests_properties(
      "${_proc_id}/${_test_name}"
      PROPERTIES
        LABELS "${_proc_id};${_test_name}"
        TIMEOUT "${XVEC_TEST_TIMEOUT}"
    )

    list(APPEND _xvec_all_test_targets "${_target}")
  endforeach()
endforeach()

# ── Convenience check target ──────────────────────────────────────────────────
# Rebuilds every selected test executable and runs the full configured suite.
#
# Usage:
#   cmake --build <build-dir> --target check
if(_xvec_all_test_targets)
  add_custom_target(
    check
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
      --output-on-failure
      --parallel "${XVEC_NUM_PARALLEL_TEST_JOBS}"
    DEPENDS
      ${_xvec_all_test_targets}
    WORKING_DIRECTORY
      "${CMAKE_BINARY_DIR}"
    USES_TERMINAL
    VERBATIM
  )
endif()

# ── Configuration summary ─────────────────────────────────────────────────────
list(LENGTH _xvec_test_sources _xvec_test_source_count)
list(LENGTH _xvec_processor_ids _xvec_processor_count)
list(LENGTH _xvec_all_test_targets _xvec_test_target_count)

message(
  STATUS
  "xvec: compiler=${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})"
)
message(STATUS "xvec: C++ standard=${CMAKE_CXX_STANDARD}")
message(STATUS "xvec: Intel SDE=${XVEC_SDE_EXECUTABLE}")
message(STATUS "xvec: test scope=${XVEC_TEST_TYPE}")
message(STATUS "xvec: processor targets=${_xvec_processor_ids}")
message(STATUS "xvec: processors=${_xvec_processor_count}")
message(STATUS "xvec: test sources=${_xvec_test_source_count}")
message(STATUS "xvec: test executables=${_xvec_test_target_count}")
message(STATUS "xvec: unit-test workflow:")
message(STATUS "      cmake --build <build-dir> --parallel")
message(
  STATUS
  "      ctest --test-dir <build-dir> --output-on-failure"
)
