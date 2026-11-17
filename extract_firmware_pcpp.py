#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# The pcpp-based half of extract-firmware-nova.py: everything it does to read a constant
# out of RM's C source.  Two things are asked of it:
#
#   parse_c_value()  evaluates the C constant expression that NVOC assigned to a Hal field
#   macro_value()    returns the value of an object-like "#define" in an RM header
#
# Keeping it here rather than in the script means pcpp is reached through one module, and
# that the script itself needs to know nothing about it.

import copy
import sys

from extract_firmware_common import MyException

try:
    from pcpp import Action, Evaluator, OutputDirective, Preprocessor
except ImportError:
    print("Error: python3-pcpp package is not installed", file=sys.stderr)
    sys.exit(1)

# pcpp's evaluator holds no state between calls, so one instance serves every call.  It
# returns an int subclass that additionally tracks whether the value is unsigned.
c_expression_evaluator = Evaluator()

# Parse a C integer constant expression and return its value.  Handles plain constants,
# hex ("0x..."), and arbitrary arithmetic, bitwise, and conditional expressions, such as
# "((22U << 20) + (32U << 20) + ((48U << 10) * 2048U))" or the expansion of an NV_MAX,
# "(((17485824) > (17482752)) ? (17485824) : (17482752)) + 4096U".  Every name in the
# expression must already be expanded, apart from the NvBool constants handled below.
def parse_c_value(value: str) -> int | bool:
    # Make a guess as to whether the expression is boolean or not.  This is imperfect,
    # and may need to be changed for future versions of OpenRM, because booleans and
    # integers are interchangeable in C.  For now, assume that any expression with
    # boolean macros in it is also boolean.  This breaks on ternary expressions like
    # "NV_TRUE ? x : y" however.
    is_bool = any(s in value for s in ("(NvBool)", "NV_TRUE", "NV_FALSE"))
    expression = value.replace("(NvBool)", "")

    try:
        result = c_expression_evaluator(expression,
                                        identifiers = {"NV_TRUE": 1, "NV_FALSE": 0}).value()
    except SyntaxError as e:
        raise MyException(f"Cannot evaluate C expression '{value}': {e}")

    # This should catch expressions where we miscalculate is_bool.
    if is_bool and result not in (0, 1):
        raise MyException(f"Invalid boolean expression '{value}'")

    return bool(result) if is_bool else result

# RM headers guard many of their constants with "#if" blocks that select between build
# configurations, so a header is read by running it through pcpp's C preprocessor rather
# than by pattern matching its text.  The preprocessor evaluates those conditionals the way
# a compiler does, so only the definitions from the branch that is actually taken survive.
class HeaderPreprocessor(Preprocessor):
    # Headers that are not on the search path are dropped rather than reported as errors,
    # because most of what an RM header includes is irrelevant here.  The macros such a
    # header would have defined are then simply undefined, and since an undefined name
    # evaluates to 0 in an "#if", the result is the branch that a build with the feature
    # turned off would take.
    def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath):
        raise OutputDirective(Action.IgnoreAndRemove)

    # The default prints to stderr and keeps going with a bad parse.
    def on_error(self, file, line, msg):
        raise MyException(f"{file}:{line}: {msg}")

# Headers that have already been preprocessed, keyed by file name, so that the repeated
# lookups parse_legacy_hals() makes in one header only preprocess it once.
preprocessed_headers = {}

# Preprocess a header and return the Preprocessor, whose "macros" dictionary then holds
# every macro that is still defined at the end of the file.
def preprocess_header(filename: str) -> Preprocessor:
    if filename in preprocessed_headers:
        return preprocessed_headers[filename]

    pp = HeaderPreprocessor()

    # RM headers include each other relative to these two directories.  Resolving those
    # includes is what gives a conditional such as "#if RMCFG_FEATURE_GSPRM_BULLSEYE" the
    # value the driver was configured with instead of the assumed default of 0.
    pp.add_path("src/nvidia/generated")
    pp.add_path("src/nvidia/inc")

    with open(filename) as f:
        pp.parse(f.read(), filename)

    # parse() only sets up the token stream.  The directives in it are executed as the
    # stream is consumed, so it has to be drained before "macros" is complete.  The
    # preprocessed text itself is of no interest.
    while pp.token():
        pass

    preprocessed_headers[filename] = pp
    return pp

# Look up an object-like "#define <macro> <expression>" in a C header and return the
# evaluated expression.  Definitions are often annotated with a trailing comment, as in
#
#   #define GSP_FW_HEAP_PARAM_BASE_RM_SIZE_GR10X              (32 << 20)   // GR10X+
#
# and long ones are continued across lines with a trailing backslash.  The preprocessor
# strips both, so the body reaches parse_c_value() as a bare expression.
#
# A definition may also name other macros, as in
#
#   #define GSP_FW_HEAP_PARAM_MAX_SUPPORTED_CHANNELS  2048   // Support 2048 channels
#   #define GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE      ((48 << 10) * GSP_FW_HEAP_PARAM_MAX_SUPPORTED_CHANNELS)
#
# Those names are expanded down to a literal expression first, so lookups can be made in
# any order, and an older header that spells the same value out in full needs no special
# handling.
def macro_value(filename: str, macro: str, fallback: str = None) -> int | bool:
    pp = preprocess_header(filename)

    if macro not in pp.macros and fallback:
        macro = fallback

    if macro not in pp.macros:
        raise MyException(f"Could not find {macro} in {filename}")

    definition = pp.macros[macro]

    # Only object-like macros have a value that can stand on its own.
    if definition.arglist is not None:
        raise MyException(f"{macro} in {filename} is a function-like macro")

    # expand_macros() rewrites the tokens it is handed in place, so give it copies and
    # leave the stored definition intact for the next lookup.
    body = pp.expand_macros([copy.copy(token) for token in definition.value])

    return parse_c_value("".join(token.value for token in body))
