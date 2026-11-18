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
# Everything extract-firmware-nova.py does to read a constant out of RM's C source.  Two
# things are asked of it:
#
#   parse_c_value()  evaluates the C constant expression that NVOC assigned to a Hal field
#   macro_value()    returns the value of an object-like "#define" in an RM header
#
# Only the second one needs pcpp.  NVOC pastes Hal field expressions into the generated .c
# file with every macro already expanded, so an expression such as
#
#   (((((4U << 20)) + (0U)) + ((((64U * 1024U))) - 1)) & ~(...))
#
# is nothing but integer arithmetic, and the evaluator below does it unaided.  That is what
# lets an r630 or newer tree, which carries its constants in those tables, be scraped on a
# machine where pcpp is not installed.  An r620 or older tree keeps its constants in
# headers that have to be preprocessed to be read at all, and there is no substitute for
# that, so those trees still need the package.

import copy
import re
import sys

from extract_firmware_common import MyException

# Both halves of pcpp are optional, and both are named here so that the rest of the module
# can test for them: PcppEvaluator to decide whether its evaluator can be run beside the
# one below, and HeaderPreprocessor to report a tree that needs preprocessing on a machine
# that cannot do it.  Preprocessor itself is bound so that the annotation on
# preprocess_header() still resolves when the import failed.
#
# RM headers guard many of their constants with "#if" blocks that select between build
# configurations, so a header is read by running it through pcpp's C preprocessor rather
# than by pattern matching its text.  The preprocessor evaluates those conditionals the way
# a compiler does, so only the definitions from the branch that is actually taken survive.
try:
    from pcpp import Action, Evaluator as PcppEvaluator, OutputDirective, Preprocessor

    class HeaderPreprocessor(Preprocessor):
        # Headers that are not on the search path are dropped rather than reported as
        # errors, because most of what an RM header includes is irrelevant here.  The
        # macros such a header would have defined are then simply undefined, and since an
        # undefined name evaluates to 0 in an "#if", the result is the branch that a build
        # with the feature turned off would take.
        def on_include_not_found(self, is_malformed, is_system_include, curdir,
                                 includepath):
            raise OutputDirective(Action.IgnoreAndRemove)

        # The default prints to stderr and keeps going with a bad parse.
        def on_error(self, file, line, msg):
            raise MyException(f"{file}:{line}: {msg}")
except ImportError:
    PcppEvaluator = Preprocessor = HeaderPreprocessor = None
# One token of a C constant expression.  Integer suffixes are matched but discarded, as
# the signedness they carry only matters for results this module refuses to produce (see
# Evaluator.__call__), and "space" exists so that the scanner can skip whitespace without
# a separate pass.
TOKEN = re.compile(r"""
      (?P<number>0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*
    | (?P<operator><<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!()?:<>])
    | (?P<name>[A-Za-z_]\w*)
    | (?P<space>\s+)
""", re.VERBOSE)

# The binary operators, in groups of equal precedence, loosest group first.  Walking this
# list is what gives "1 | 2 ^ 3 & 4" the grouping C gives it.  The conditional operator is
# absent because it takes three operands and associates to the right, so parse_conditional
# handles it, and the unary operators are absent because they bind tighter than all of
# these, so parse_unary handles them.
PRECEDENCE = [
    ["||"],
    ["&&"],
    ["|"],
    ["^"],
    ["&"],
    ["==", "!="],
    ["<", "<=", ">", ">="],
    ["<<", ">>"],
    ["+", "-"],
    ["*", "/", "%"],
]

# C truncates integer division toward zero, where Python's // rounds toward negative
# infinity, so the two disagree whenever exactly one operand is negative.  None of the
# expressions NVOC emits divides at all, but getting this wrong silently would be worse
# than not supporting division, so do it the way C does.
def c_divide(a: int, b: int) -> int:
    if b == 0:
        raise SyntaxError("division by zero")

    return abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)

def c_modulo(a: int, b: int) -> int:
    return a - b * c_divide(a, b)

# C's comparison and logical operators yield an int, not a bool, and the difference shows
# up once a result is used in further arithmetic, so every one of these returns an int.
APPLY = {
    "||": lambda a, b: int(bool(a) or bool(b)),
    "&&": lambda a, b: int(bool(a) and bool(b)),
    "|":  lambda a, b: a | b,
    "^":  lambda a, b: a ^ b,
    "&":  lambda a, b: a & b,
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
    "<":  lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    ">":  lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "/":  c_divide,
    "%":  c_modulo,
}

# A leading zero makes an integer constant octal in C, which int(text, 0) rejects rather
# than honouring, so the two cases are told apart here.  "0" itself is not octal.
def c_integer(text: str) -> int:
    try:
        if len(text) > 1 and text[0] == "0" and text[1] not in "xX":
            return int(text, 8)

        return int(text, 0)
    except ValueError:
        raise SyntaxError(f"invalid integer constant {text}")

# The result of an evaluation.  pcpp hands back an int subclass carrying a value() method,
# and parse_c_value() calls it, so do the same rather than make the caller care which
# evaluator answered.
class Value(int):
    def value(self) -> int:
        return int(self)

# Evaluates one C constant expression by recursive descent.  pcpp's evaluator keeps no
# state between calls and neither does this, so a single instance serves every call; the
# parsing state belongs to Expression instead.
class Evaluator:
    def __call__(self, expression: str, identifiers: dict = None) -> Value:
        result = Expression(expression, identifiers or {}).evaluate()

        # pcpp evaluates in 64-bit integers and truncates anything that does not fit,
        # while Python integers grow without bound, so the two agree only as long as a
        # result stays in range.  Every value in an NVOC Hal field table is a memory size
        # of a few hundred megabytes at most, so rather than reimplement the truncation,
        # refuse to return a result this wide: one means the expression is not the kind
        # this module was written to read, and a wrong size is worse than no size.
        if not -(1 << 63) <= result < (1 << 64):
            raise SyntaxError(f"result {result:#x} does not fit in 64 bits")

        return Value(result)

# One expression mid-parse: the token list and how far into it the parser has read.  The
# tokens are plain Python ints for integer constants and identifiers that were resolved,
# and strings for operators and parentheses, so a token being an int is what distinguishes
# an operand from everything else.
class Expression:
    def __init__(self, text: str, identifiers: dict):
        self.tokens = []
        position = 0

        while position < len(text):
            token = TOKEN.match(text, position)
            if not token:
                raise SyntaxError(f"stray '{text[position]}'")
            position = token.end()

            if token.lastgroup == "space":
                continue

            if token.lastgroup == "number":
                self.tokens.append(c_integer(token.group("number")))
            elif token.lastgroup == "name":
                name = token.group("name")
                # An undefined name is an error rather than the zero that the C
                # preprocessor would substitute in an "#if", because a Hal field value
                # that quietly became zero would ship as a real memory size.  pcpp words
                # this the same way.
                if name not in identifiers:
                    raise SyntaxError(f"Unknown identifier {name}")
                self.tokens.append(identifiers[name])
            else:
                self.tokens.append(token.group("operator"))

        self.position = 0

    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def accept(self, operator: str) -> bool:
        if self.peek() == operator:
            self.position += 1
            return True

        return False

    def expect(self, operator: str):
        if not self.accept(operator):
            raise SyntaxError(f"expected '{operator}' at {self.remainder()}")

    def remainder(self) -> str:
        return "EOF" if self.peek() is None else f"'{self.peek()}'"

    # A parenthesised subexpression, an integer constant, or a unary operator applied to
    # one of those.
    def parse_unary(self) -> int:
        token = self.peek()
        if token is None:
            raise SyntaxError("at EOF")
        self.position += 1

        if token == "(":
            value = self.parse_conditional()
            self.expect(")")
            return value

        if token == "-":
            return -self.parse_unary()
        if token == "+":
            return self.parse_unary()
        if token == "~":
            return ~self.parse_unary()
        if token == "!":
            return int(not self.parse_unary())

        if isinstance(token, int):
            return token

        raise SyntaxError(f"unexpected '{token}'")

    # Every binary operator at `level` and tighter, left associative as C has them.
    def parse_binary(self, level: int = 0) -> int:
        if level == len(PRECEDENCE):
            return self.parse_unary()

        value = self.parse_binary(level + 1)

        while self.peek() in PRECEDENCE[level]:
            operator = self.tokens[self.position]
            self.position += 1
            value = APPLY[operator](value, self.parse_binary(level + 1))

        return value

    # "condition ? yes : no", which associates to the right, so both arms parse as whole
    # conditional expressions and a nested "a ? b ? c : d : e" comes out the way C reads it.
    def parse_conditional(self) -> int:
        condition = self.parse_binary()

        if not self.accept("?"):
            return condition

        yes = self.parse_conditional()
        self.expect(":")
        no = self.parse_conditional()

        return yes if condition else no

    def evaluate(self) -> int:
        value = self.parse_conditional()

        # Anything left over means the text was not one expression, which usually means a
        # caller handed over something it had not finished stripping, such as a cast.
        if self.position != len(self.tokens):
            raise SyntaxError(f"unexpected {self.remainder()} after expression")

        return value
# Neither evaluator holds state between calls, so one instance of each serves every call.
# Both are run on every expression whenever both are installed, and pcpp goes first so
# that its answer is the one used.  Running the stand-in alongside it rather than instead
# of it is what verifies the stand-in against the tree being scraped: an expression a new
# OpenRM has started emitting that the stand-in reads differently, or cannot read at all,
# then stops a run here, instead of waiting to be found on a machine that has no pcpp and
# therefore nothing to compare against.
c_expression_evaluators = [evaluator() for evaluator in (PcppEvaluator, Evaluator)
                           if evaluator]

# NVOC writes booleans as "((NvBool)(0 == 0))", but older trees name the NvBool constants
# directly, so the evaluators are told what they stand for.
nvbool_identifiers = {"NV_TRUE": 1, "NV_FALSE": 0}

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

    results = {}
    for evaluator in c_expression_evaluators:
        who = type(evaluator).__module__.split(".")[0]
        try:
            results[who] = evaluator(expression, identifiers = nvbool_identifiers).value()
        except SyntaxError as e:
            raise MyException(f"{who} cannot evaluate C expression '{value}': {e}")

    if len(set(results.values())) > 1:
        raise MyException(f"Evaluators disagree on C expression '{value}': {results}")

    result = next(iter(results.values()))

    # This should catch expressions where we miscalculate is_bool.
    if is_bool and result not in (0, 1):
        raise MyException(f"Invalid boolean expression '{value}'")

    return bool(result) if is_bool else result
# Headers that have already been preprocessed, keyed by file name, so that the repeated
# lookups parse_legacy_hals() makes in one header only preprocess it once.
preprocessed_headers = {}

# Preprocess a header and return the Preprocessor, whose "macros" dictionary then holds
# every macro that is still defined at the end of the file.
def preprocess_header(filename: str) -> Preprocessor:
    if filename in preprocessed_headers:
        return preprocessed_headers[filename]

    # This is the only thing the package is still needed for, so it is also the only place
    # that has to notice it is missing.  Naming the header says which constant was wanted.
    if HeaderPreprocessor is None:
        raise MyException(f"Reading the constants in {filename} requires the "
                          f"python3-pcpp package, which is not installed")

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
