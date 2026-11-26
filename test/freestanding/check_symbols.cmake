# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

execute_process(
  COMMAND "${NM}" -u "${OBJECT}"
  COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_VARIABLE undefined_symbols
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT undefined_symbols STREQUAL "")
  message(FATAL_ERROR "Freestanding probe has undefined symbols:\n${undefined_symbols}")
endif()
