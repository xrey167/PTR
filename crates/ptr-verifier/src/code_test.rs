//! Running a candidate program against the task's tests, in an embedded sandboxed interpreter.
//!
//! # Why an embedded interpreter
//!
//! `ptr-exec` offers isolates (state machines with bounded mailboxes) but no process isolation,
//! and the workspace embeds no general-purpose language. Running a candidate under an arbitrary
//! interpreter would hand it the host's filesystem and network. So the candidate is written in a
//! tiny language defined here, whose programs *cannot* touch anything: the only values are `i64`
//! integers, the only effects are returning a value and `print`ing into a bounded buffer, and
//! there is no I/O, no host call, no clock and no randomness. Execution is a pure function of the
//! source and the arguments, so a report is reproducible.
//!
//! # The language
//!
//! ```text
//! program := fn*
//! fn      := "fn" name "(" [name ("," name)*] ")" block
//! stmt    := "let" name "=" expr ";" | name "=" expr ";" | "return" expr ";" | "print" expr ";"
//!          | "if" expr block ["else" (block | if)] | "while" expr block
//! expr    := integers, names, calls "f(a, b)", "(" expr ")", unary "-" "!",
//!            "* / %", "+ -", "< <= > >=", "== !=", "&&", "||"      (booleans are 0 and 1)
//! ```
//!
//! `//` starts a comment. Variables are function-scoped. Arithmetic is checked: overflow and
//! division by zero are runtime errors of that test.
//!
//! # Bounds (all hard, all findings, none a panic)
//!
//! * source at most [`MAX_SOURCE_BYTES`], at most [`MAX_TOKENS`] tokens and [`MAX_FUNCTIONS`]
//!   functions, nesting at most [`MAX_NESTING`] (`code.oversize` / `code.syntax`);
//! * CPU time: [`FUEL_PER_TEST`] fuel per test (`code.timeout`). Every statement, expression
//!   node and loop iteration costs 1, and a call additionally costs the number of distinct
//!   variable names of the callee (its frame size). Variable names are resolved to frame slot
//!   indices when the program is parsed, so each step is `O(1)` (calls to a function are looked
//!   up by name in a map of at most [`MAX_FUNCTIONS`]). The worst case of one `verify` is
//!   therefore about [`FUEL_PER_TEST`] x [`MAX_TESTS`] = 3.2 million elementary steps plus the
//!   bounded print buffer: fuel is a deterministic bound on work, not a wall-clock limit;
//! * host stack: native recursion at most [`MAX_NATIVE_DEPTH`] frames in total, so the three
//!   nesting bounds below cannot multiply into a stack overflow (`code.resource`); a test in
//!   this module runs the deepest admitted program on a 512 KiB thread in a debug build;
//! * memory: at most [`MAX_LOCALS`] live variables over all frames, call depth at most
//!   [`MAX_CALL_DEPTH`] (`code.resource`);
//! * output: at most [`MAX_OUTPUT_BYTES`] printed bytes per test (`code.resource`).
//!
//! # Test visibility
//!
//! A [`CodeTest`] is [`TestVisibility::Hidden`] by default: a failing test's finding names the
//! test id and nothing else, so a repair Pod cannot read the expected value (or a runtime error
//! detail) out of the message and hard-code it. A [`TestVisibility::Visible`] test (the task's
//! examples) reports `expected X, got Y` and the runtime error. Timeout and resource findings
//! carry no test data and are the same for both.
//!
//! The report passes only if every test returns the expected value (and printed output, when the
//! test states one). A task with no tests, too many tests or a bad test id is `Unknown`
//! (`code.task`). See [`crate::codes`].

use std::collections::BTreeMap;

use ptr_types::{Probability, VerificationLevel};

use crate::codes::finding;
use crate::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// Longest candidate source, in bytes.
pub const MAX_SOURCE_BYTES: usize = 8 * 1024;
/// Most tokens in a candidate.
pub const MAX_TOKENS: usize = 2048;
/// Most functions in a candidate.
pub const MAX_FUNCTIONS: usize = 32;
/// Deepest nesting of blocks and expressions.
pub const MAX_NESTING: usize = 16;
/// Evaluation steps one test may take.
pub const FUEL_PER_TEST: u64 = 100_000;
/// Most variables alive at once, over all call frames.
pub const MAX_LOCALS: usize = 256;
/// Deepest call stack.
pub const MAX_CALL_DEPTH: usize = 20;
/// Deepest native recursion of the interpreter (nested blocks and expression nodes, summed over
/// all live calls). It bounds the host stack the interpreter can use whatever combination of
/// call depth, block nesting and expression height a program builds; reaching it is
/// `code.resource`.
pub const MAX_NATIVE_DEPTH: usize = 256;
/// Most bytes one test may print.
pub const MAX_OUTPUT_BYTES: usize = 1024;
/// Most tests in a task.
pub const MAX_TESTS: usize = 32;
/// Longest test id, in bytes.
pub const MAX_TEST_ID_BYTES: usize = 64;

/// One test of a task: call the entry function with `args`, expect `expected`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodeTest {
    pub id: String,
    pub args: Vec<i64>,
    pub expected: i64,
    /// What the call must print, if the test checks it.
    pub expected_output: Option<String>,
    /// What a failure message may reveal. [`TestVisibility::Hidden`] unless set otherwise.
    pub visibility: TestVisibility,
}

/// What the finding of a failing test may say about it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TestVisibility {
    /// The message names the test id only (the default, safe for held-out tests).
    #[default]
    Hidden,
    /// The message may state the expected and the actual value, or the runtime error.
    Visible,
}

impl CodeTest {
    pub fn new(id: &str, args: &[i64], expected: i64) -> Self {
        Self {
            id: id.to_owned(),
            args: args.to_vec(),
            expected,
            expected_output: None,
            visibility: TestVisibility::Hidden,
        }
    }

    /// The same test with its failure details visible in the finding.
    pub fn visible(mut self) -> Self {
        self.visibility = TestVisibility::Visible;
        self
    }
}

/// The task: the entry function and the tests the candidate must pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodeTask {
    pub entry: String,
    pub tests: Vec<CodeTest>,
}

// ------------------------------------------------------------------------------------------
// Lexer
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
enum Tok {
    Int(i64),
    Name(String),
    Sym(&'static str),
}

/// Two-character symbols first, so `==` is never read as two `=`.
const SYMBOLS: [&str; 21] = [
    "==", "!=", "<=", ">=", "&&", "||", "+", "-", "*", "/", "%", "<", ">", "!", "=", "(", ")", "{",
    "}", ",", ";",
];

struct SyntaxError {
    line: usize,
    why: &'static str,
}

fn lex(source: &str) -> Result<Vec<(Tok, usize)>, SyntaxError> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut pos = 0;
    let mut line = 1;
    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte == b'\n' {
            line += 1;
            pos += 1;
        } else if byte.is_ascii_whitespace() {
            pos += 1;
        } else if bytes[pos..].starts_with(b"//") {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
        } else {
            if tokens.len() >= MAX_TOKENS {
                return Err(SyntaxError {
                    line,
                    why: "too many tokens",
                });
            }
            if byte.is_ascii_digit() {
                let start = pos;
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
                let value = source[start..pos].parse::<i64>().map_err(|_| SyntaxError {
                    line,
                    why: "integer literal out of range",
                })?;
                tokens.push((Tok::Int(value), line));
            } else if byte.is_ascii_alphabetic() || byte == b'_' {
                let start = pos;
                while pos < bytes.len()
                    && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_')
                {
                    pos += 1;
                }
                tokens.push((Tok::Name(source[start..pos].to_owned()), line));
            } else {
                let Some(symbol) = SYMBOLS
                    .iter()
                    .find(|s| bytes[pos..].starts_with(s.as_bytes()))
                else {
                    return Err(SyntaxError {
                        line,
                        why: "unexpected character",
                    });
                };
                pos += symbol.len();
                tokens.push((Tok::Sym(symbol), line));
            }
        }
    }
    Ok(tokens)
}

// ------------------------------------------------------------------------------------------
// AST and parser
// ------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Debug)]
enum ExprKind {
    Int(i64),
    Var(usize),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug)]
struct Expr {
    /// Height of the tree; bounded so evaluation recursion is bounded.
    height: usize,
    kind: ExprKind,
}

#[derive(Debug)]
enum Stmt {
    Let(usize, Expr),
    Assign(usize, Expr),
    Return(Expr),
    Print(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
}

#[derive(Debug)]
struct Function {
    param_count: usize,
    /// Distinct variable names of the function (parameters first): the frame size.
    slots: usize,
    body: Vec<Stmt>,
}

struct Parser {
    tokens: Vec<(Tok, usize)>,
    pos: usize,
    /// Names of the function being parsed, resolved to frame slot indices.
    scope: BTreeMap<String, usize>,
}

type ParseResult<T> = Result<T, SyntaxError>;

const KEYWORDS: [&str; 7] = ["fn", "let", "if", "else", "while", "return", "print"];

impl Parser {
    fn line(&self) -> usize {
        self.tokens
            .get(self.pos)
            .or_else(|| self.tokens.last())
            .map_or(1, |(_, line)| *line)
    }

    fn fail<T>(&self, why: &'static str) -> ParseResult<T> {
        Err(SyntaxError {
            line: self.line(),
            why,
        })
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos).map(|(tok, _)| tok)
    }

    fn eat_sym(&mut self, symbol: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(s)) if *s == symbol) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_sym(&mut self, symbol: &str, why: &'static str) -> ParseResult<()> {
        if self.eat_sym(symbol) {
            Ok(())
        } else {
            self.fail(why)
        }
    }

    fn eat_keyword(&mut self, keyword: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Name(n)) if n == keyword) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn name(&mut self, why: &'static str) -> ParseResult<String> {
        match self.peek() {
            Some(Tok::Name(name)) if !KEYWORDS.contains(&name.as_str()) => {
                let name = name.clone();
                self.pos += 1;
                Ok(name)
            }
            _ => self.fail(why),
        }
    }

    fn slot(&mut self, name: String) -> usize {
        let next = self.scope.len();
        *self.scope.entry(name).or_insert(next)
    }

    fn program(&mut self) -> ParseResult<BTreeMap<String, Function>> {
        let mut functions = BTreeMap::new();
        while self.peek().is_some() {
            if !self.eat_keyword("fn") {
                return self.fail("expected `fn`");
            }
            let name = self.name("expected a function name")?;
            self.expect_sym("(", "expected `(`")?;
            self.scope.clear();
            let mut param_count = 0;
            if !self.eat_sym(")") {
                loop {
                    let param = self.name("expected a parameter name")?;
                    if self.scope.contains_key(&param) {
                        return self.fail("duplicate parameter");
                    }
                    self.slot(param);
                    param_count += 1;
                    if self.eat_sym(")") {
                        break;
                    }
                    self.expect_sym(",", "expected `,` or `)`")?;
                }
            }
            let body = self.block(0)?;
            if functions.len() >= MAX_FUNCTIONS {
                return self.fail("too many functions");
            }
            let function = Function {
                param_count,
                slots: self.scope.len(),
                body,
            };
            if functions.insert(name, function).is_some() {
                return self.fail("function defined twice");
            }
        }
        Ok(functions)
    }

    fn block(&mut self, depth: usize) -> ParseResult<Vec<Stmt>> {
        if depth >= MAX_NESTING {
            return self.fail("blocks nested too deeply");
        }
        self.expect_sym("{", "expected `{`")?;
        let mut statements = Vec::new();
        while !self.eat_sym("}") {
            if self.peek().is_none() {
                return self.fail("unterminated block");
            }
            statements.push(self.statement(depth)?);
        }
        Ok(statements)
    }

    fn statement(&mut self, depth: usize) -> ParseResult<Stmt> {
        if self.eat_keyword("let") {
            let name = self.name("expected a variable name")?;
            self.expect_sym("=", "expected `=`")?;
            let value = self.expr(0)?;
            self.expect_sym(";", "expected `;`")?;
            let slot = self.slot(name);
            Ok(Stmt::Let(slot, value))
        } else if self.eat_keyword("return") {
            let value = self.expr(0)?;
            self.expect_sym(";", "expected `;`")?;
            Ok(Stmt::Return(value))
        } else if self.eat_keyword("print") {
            let value = self.expr(0)?;
            self.expect_sym(";", "expected `;`")?;
            Ok(Stmt::Print(value))
        } else if self.eat_keyword("while") {
            let condition = self.expr(0)?;
            Ok(Stmt::While(condition, self.block(depth + 1)?))
        } else if self.eat_keyword("if") {
            let condition = self.expr(0)?;
            let then = self.block(depth + 1)?;
            let otherwise = if self.eat_keyword("else") {
                if matches!(self.peek(), Some(Tok::Name(n)) if n == "if") {
                    if depth + 1 >= MAX_NESTING {
                        return self.fail("blocks nested too deeply");
                    }
                    vec![self.statement(depth + 1)?]
                } else {
                    self.block(depth + 1)?
                }
            } else {
                Vec::new()
            };
            Ok(Stmt::If(condition, then, otherwise))
        } else {
            let name = self.name("expected a statement")?;
            self.expect_sym("=", "expected `=`")?;
            let value = self.expr(0)?;
            self.expect_sym(";", "expected `;`")?;
            let slot = self.slot(name);
            Ok(Stmt::Assign(slot, value))
        }
    }

    fn node(&self, kind: ExprKind, height: usize) -> ParseResult<Expr> {
        if height > MAX_NESTING * 2 {
            return self.fail("expression too deep");
        }
        Ok(Expr { height, kind })
    }

    fn binary(&mut self, depth: usize, level: usize) -> ParseResult<Expr> {
        const LEVELS: [&[(&str, BinOp)]; 6] = [
            &[("||", BinOp::Or)],
            &[("&&", BinOp::And)],
            &[("==", BinOp::Eq), ("!=", BinOp::Ne)],
            &[
                ("<=", BinOp::Le),
                (">=", BinOp::Ge),
                ("<", BinOp::Lt),
                (">", BinOp::Gt),
            ],
            &[("+", BinOp::Add), ("-", BinOp::Sub)],
            &[("*", BinOp::Mul), ("/", BinOp::Div), ("%", BinOp::Rem)],
        ];
        if level == LEVELS.len() {
            return self.unary(depth);
        }
        let mut left = self.binary(depth, level + 1)?;
        loop {
            let Some(&(_, op)) = LEVELS[level]
                .iter()
                .find(|(symbol, _)| matches!(self.peek(), Some(Tok::Sym(s)) if s == symbol))
            else {
                return Ok(left);
            };
            self.pos += 1;
            let right = self.binary(depth, level + 1)?;
            let height = 1 + left.height.max(right.height);
            left = self.node(ExprKind::Bin(op, Box::new(left), Box::new(right)), height)?;
        }
    }

    fn expr(&mut self, depth: usize) -> ParseResult<Expr> {
        if depth >= MAX_NESTING {
            return self.fail("expression nested too deeply");
        }
        self.binary(depth, 0)
    }

    fn unary(&mut self, depth: usize) -> ParseResult<Expr> {
        if depth >= MAX_NESTING {
            return self.fail("expression nested too deeply");
        }
        if self.eat_sym("-") {
            let inner = self.unary(depth + 1)?;
            let height = inner.height + 1;
            return self.node(ExprKind::Neg(Box::new(inner)), height);
        }
        if self.eat_sym("!") {
            let inner = self.unary(depth + 1)?;
            let height = inner.height + 1;
            return self.node(ExprKind::Not(Box::new(inner)), height);
        }
        self.primary(depth)
    }

    fn primary(&mut self, depth: usize) -> ParseResult<Expr> {
        match self.peek().cloned() {
            Some(Tok::Int(value)) => {
                self.pos += 1;
                self.node(ExprKind::Int(value), 1)
            }
            Some(Tok::Sym("(")) => {
                self.pos += 1;
                let inner = self.expr(depth + 1)?;
                self.expect_sym(")", "expected `)`")?;
                Ok(inner)
            }
            Some(Tok::Name(name)) if !KEYWORDS.contains(&name.as_str()) => {
                self.pos += 1;
                if self.eat_sym("(") {
                    let mut args = Vec::new();
                    if !self.eat_sym(")") {
                        loop {
                            args.push(self.expr(depth + 1)?);
                            if self.eat_sym(")") {
                                break;
                            }
                            self.expect_sym(",", "expected `,` or `)`")?;
                        }
                    }
                    if args.len() > 8 {
                        return self.fail("too many arguments");
                    }
                    let height = 1 + args.iter().map(|a| a.height).max().unwrap_or(0);
                    self.node(ExprKind::Call(name, args), height)
                } else {
                    let slot = self.slot(name);
                    self.node(ExprKind::Var(slot), 1)
                }
            }
            _ => self.fail("expected an expression"),
        }
    }
}

fn parse(source: &str) -> Result<BTreeMap<String, Function>, SyntaxError> {
    if !source.is_ascii() {
        return Err(SyntaxError {
            line: 1,
            why: "source must be ASCII",
        });
    }
    let tokens = lex(source)?;
    Parser {
        tokens,
        pos: 0,
        scope: BTreeMap::new(),
    }
    .program()
}

// ------------------------------------------------------------------------------------------
// Interpreter
// ------------------------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
enum Trap {
    Timeout,
    Resource(&'static str),
    Runtime(&'static str),
}

struct Machine<'p> {
    functions: &'p BTreeMap<String, Function>,
    fuel: u64,
    /// All frames, one `Option` per variable slot (`None` until its first `let`).
    locals: Vec<Option<i64>>,
    /// Number of defined variables over all frames (bounded by [`MAX_LOCALS`]).
    live: usize,
    depth: usize,
    /// Native recursion in progress (see [`MAX_NATIVE_DEPTH`]).
    native: usize,
    output: String,
}

impl Machine<'_> {
    fn charge(&mut self, amount: u64) -> Result<(), Trap> {
        self.fuel = self.fuel.checked_sub(amount).ok_or(Trap::Timeout)?;
        Ok(())
    }

    fn tick(&mut self) -> Result<(), Trap> {
        self.charge(1)
    }

    fn call(&mut self, name: &str, args: &[i64]) -> Result<i64, Trap> {
        let function = self
            .functions
            .get(name)
            .ok_or(Trap::Runtime("call of an undefined function"))?;
        if function.param_count != args.len() {
            return Err(Trap::Runtime("wrong number of arguments"));
        }
        if self.depth >= MAX_CALL_DEPTH {
            return Err(Trap::Resource("call depth bound exceeded"));
        }
        if self.live + args.len() > MAX_LOCALS {
            return Err(Trap::Resource("memory bound exceeded"));
        }
        // Setting up the frame costs its size, so work stays bounded by fuel.
        self.charge(function.slots as u64)?;
        let base = self.locals.len();
        self.locals.resize(base + function.slots, None);
        for (cell, value) in self.locals[base..].iter_mut().zip(args) {
            *cell = Some(*value);
        }
        self.live += args.len();
        self.depth += 1;
        let result = self.block(&function.body, base);
        self.depth -= 1;
        self.live -= self.locals[base..].iter().flatten().count();
        self.locals.truncate(base);
        match result? {
            Some(value) => Ok(value),
            None => Err(Trap::Runtime("function ended without returning")),
        }
    }

    fn enter(&mut self) -> Result<(), Trap> {
        if self.native >= MAX_NATIVE_DEPTH {
            return Err(Trap::Resource("native recursion bound exceeded"));
        }
        self.native += 1;
        Ok(())
    }

    fn block(&mut self, statements: &[Stmt], base: usize) -> Result<Option<i64>, Trap> {
        self.enter()?;
        let result = self.run_block(statements, base);
        self.native -= 1;
        result
    }

    fn run_block(&mut self, statements: &[Stmt], base: usize) -> Result<Option<i64>, Trap> {
        for statement in statements {
            if let Some(value) = self.statement(statement, base)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn read(&self, slot: usize, base: usize) -> Option<i64> {
        self.locals[base + slot]
    }

    #[inline(never)]
    fn print(&mut self, value: i64) -> Result<(), Trap> {
        let text = format!("{value}\n");
        if self.output.len() + text.len() > MAX_OUTPUT_BYTES {
            return Err(Trap::Resource("output bound exceeded"));
        }
        self.output.push_str(&text);
        Ok(())
    }

    fn statement(&mut self, statement: &Stmt, base: usize) -> Result<Option<i64>, Trap> {
        self.tick()?;
        match statement {
            Stmt::Let(slot, value) => {
                let value = self.eval(value, base)?;
                if self.read(*slot, base).is_none() {
                    if self.live >= MAX_LOCALS {
                        return Err(Trap::Resource("memory bound exceeded"));
                    }
                    self.live += 1;
                }
                self.locals[base + slot] = Some(value);
                Ok(None)
            }
            Stmt::Assign(slot, value) => {
                let value = self.eval(value, base)?;
                if self.read(*slot, base).is_none() {
                    return Err(Trap::Runtime("assignment to an undeclared variable"));
                }
                self.locals[base + slot] = Some(value);
                Ok(None)
            }
            Stmt::Return(value) => Ok(Some(self.eval(value, base)?)),
            Stmt::Print(value) => {
                let value = self.eval(value, base)?;
                self.print(value)?;
                Ok(None)
            }
            Stmt::If(condition, then, otherwise) => {
                if self.eval(condition, base)? != 0 {
                    self.block(then, base)
                } else {
                    self.block(otherwise, base)
                }
            }
            Stmt::While(condition, body) => {
                while self.eval(condition, base)? != 0 {
                    self.tick()?;
                    if let Some(value) = self.block(body, base)? {
                        return Ok(Some(value));
                    }
                }
                Ok(None)
            }
        }
    }

    fn eval_call(&mut self, name: &str, args: &[Expr], base: usize) -> Result<i64, Trap> {
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            values.push(self.eval(arg, base)?);
        }
        self.call(name, &values)
    }

    fn eval(&mut self, expr: &Expr, base: usize) -> Result<i64, Trap> {
        self.enter()?;
        let result = self.eval_node(expr, base);
        self.native -= 1;
        result
    }

    fn eval_node(&mut self, expr: &Expr, base: usize) -> Result<i64, Trap> {
        self.tick()?;
        match &expr.kind {
            ExprKind::Int(value) => Ok(*value),
            ExprKind::Var(slot) => self
                .read(*slot, base)
                .ok_or(Trap::Runtime("use of an undeclared variable")),
            ExprKind::Neg(inner) => self
                .eval(inner, base)?
                .checked_neg()
                .ok_or(Trap::Runtime("integer overflow")),
            ExprKind::Not(inner) => Ok(i64::from(self.eval(inner, base)? == 0)),
            ExprKind::Call(name, args) => self.eval_call(name, args, base),
            ExprKind::Bin(BinOp::And, left, right) => {
                if self.eval(left, base)? == 0 {
                    Ok(0)
                } else {
                    Ok(i64::from(self.eval(right, base)? != 0))
                }
            }
            ExprKind::Bin(BinOp::Or, left, right) => {
                if self.eval(left, base)? != 0 {
                    Ok(1)
                } else {
                    Ok(i64::from(self.eval(right, base)? != 0))
                }
            }
            ExprKind::Bin(op, left, right) => {
                let (a, b) = (self.eval(left, base)?, self.eval(right, base)?);
                apply(*op, a, b)
            }
        }
    }
}

/// A strict binary operator. Kept out of `eval` so the recursive frame stays small.
#[inline(never)]
fn apply(op: BinOp, a: i64, b: i64) -> Result<i64, Trap> {
    let overflow = Trap::Runtime("integer overflow");
    match op {
        BinOp::Add => a.checked_add(b).ok_or(overflow),
        BinOp::Sub => a.checked_sub(b).ok_or(overflow),
        BinOp::Mul => a.checked_mul(b).ok_or(overflow),
        BinOp::Div | BinOp::Rem if b == 0 => Err(Trap::Runtime("division by zero")),
        BinOp::Div => a.checked_div(b).ok_or(overflow),
        BinOp::Rem => a.checked_rem(b).ok_or(overflow),
        BinOp::Lt => Ok(i64::from(a < b)),
        BinOp::Le => Ok(i64::from(a <= b)),
        BinOp::Gt => Ok(i64::from(a > b)),
        BinOp::Ge => Ok(i64::from(a >= b)),
        BinOp::Eq => Ok(i64::from(a == b)),
        BinOp::Ne => Ok(i64::from(a != b)),
        BinOp::And | BinOp::Or => Err(Trap::Runtime("unreachable operator")),
    }
}

fn run_test(
    functions: &BTreeMap<String, Function>,
    entry: &str,
    test: &CodeTest,
) -> Result<(i64, String), Trap> {
    let mut machine = Machine {
        functions,
        fuel: FUEL_PER_TEST,
        locals: Vec::new(),
        live: 0,
        depth: 0,
        native: 0,
        output: String::new(),
    };
    let value = machine.call(entry, &test.args)?;
    Ok((value, machine.output))
}

// ------------------------------------------------------------------------------------------
// Verifier
// ------------------------------------------------------------------------------------------

/// Verifies a candidate program against the task's tests. See the module documentation.
#[derive(Clone, Debug)]
pub struct CodeTestVerifier {
    task: CodeTask,
    task_problem: Option<&'static str>,
}

impl CodeTestVerifier {
    pub fn new(task: CodeTask) -> Self {
        let mut ids = std::collections::BTreeSet::new();
        let task_problem =
            if task.tests.is_empty() {
                Some("the task has no tests")
            } else if task.tests.len() > MAX_TESTS {
                Some("the task has too many tests")
            } else if task.entry.is_empty() || task.entry.len() > MAX_TEST_ID_BYTES {
                Some("the entry name is empty or too long")
            } else if task.tests.iter().any(|t| {
                t.id.is_empty() || t.id.len() > MAX_TEST_ID_BYTES || !ids.insert(t.id.clone())
            }) {
                Some("a test id is empty, too long or repeated")
            } else if task.tests.iter().any(|t| {
                t.args.len() > 8
                    || t.expected_output
                        .as_ref()
                        .is_some_and(|o| o.len() > MAX_OUTPUT_BYTES)
            }) {
                Some("a test has too many arguments or too long an expected output")
            } else {
                None
            };
        Self { task, task_problem }
    }

    pub fn task(&self) -> &CodeTask {
        &self.task
    }
}

fn report(status: VerificationStatus, findings: Vec<Finding>) -> VerificationReport {
    let score = if status == VerificationStatus::Pass {
        1.0
    } else {
        0.0
    };
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).expect("bounded score"),
        findings,
    }
}

impl Verifier<String> for CodeTestVerifier {
    fn verify(&self, candidate: &String) -> VerificationReport {
        if let Some(problem) = self.task_problem {
            return report(
                VerificationStatus::Unknown,
                vec![finding("code.task", problem.into())],
            );
        }
        if candidate.len() > MAX_SOURCE_BYTES {
            return report(
                VerificationStatus::Fail,
                vec![finding(
                    "code.oversize",
                    format!(
                        "source is {} bytes, the bound is {MAX_SOURCE_BYTES}",
                        candidate.len()
                    ),
                )],
            );
        }
        let functions = match parse(candidate) {
            Ok(functions) => functions,
            Err(error) => {
                return report(
                    VerificationStatus::Fail,
                    vec![finding(
                        "code.syntax",
                        format!("line {}: {}", error.line, error.why),
                    )],
                );
            }
        };
        let entry = &self.task.entry;
        if !functions.contains_key(entry) {
            return report(
                VerificationStatus::Fail,
                vec![finding(
                    "code.syntax",
                    format!("entry function {entry:?} is not defined"),
                )],
            );
        }
        let mut findings = Vec::new();
        for test in &self.task.tests {
            let id = &test.id;
            match run_test(&functions, entry, test) {
                Ok((value, output)) => {
                    if value != test.expected {
                        findings.push(finding(
                            "code.test_failed",
                            match test.visibility {
                                TestVisibility::Visible => {
                                    format!("test {id}: expected {}, got {value}", test.expected)
                                }
                                TestVisibility::Hidden => format!("test {id}: failed"),
                            },
                        ));
                    } else if test
                        .expected_output
                        .as_ref()
                        .is_some_and(|expected| *expected != output)
                    {
                        findings.push(finding(
                            "code.test_failed",
                            match test.visibility {
                                TestVisibility::Visible => format!(
                                    "test {id}: printed output differs from the expected output"
                                ),
                                TestVisibility::Hidden => format!("test {id}: failed"),
                            },
                        ));
                    }
                }
                Err(Trap::Timeout) => findings.push(finding(
                    "code.timeout",
                    format!("test {id}: exceeded {FUEL_PER_TEST} evaluation steps"),
                )),
                Err(Trap::Resource(what)) => {
                    findings.push(finding("code.resource", format!("test {id}: {what}")));
                }
                Err(Trap::Runtime(what)) => findings.push(finding(
                    "code.test_failed",
                    match test.visibility {
                        TestVisibility::Visible => format!("test {id}: runtime error: {what}"),
                        TestVisibility::Hidden => format!("test {id}: failed"),
                    },
                )),
            }
        }
        let status = if findings.is_empty() {
            VerificationStatus::Pass
        } else {
            VerificationStatus::Fail
        };
        report(status, findings)
    }
}

impl NamedVerifier<String> for CodeTestVerifier {
    fn name(&self) -> &'static str {
        "code-test"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(tests: &[(&str, &[i64], i64)]) -> CodeTask {
        CodeTask {
            entry: "f".into(),
            tests: tests
                .iter()
                .map(|(id, args, expected)| CodeTest::new(id, args, *expected).visible())
                .collect(),
        }
    }

    fn check(source: &str, task: CodeTask) -> VerificationReport {
        CodeTestVerifier::new(task).verify(&source.to_owned())
    }

    fn codes(report: &VerificationReport) -> Vec<&str> {
        report.findings.iter().map(|f| f.code.as_str()).collect()
    }

    const FACTORIAL: &str = "
        fn fact(n) { if n <= 1 { return 1; } return n * fact(n - 1); }
        fn f(n) { return fact(n); }";

    fn factorial_task() -> CodeTask {
        task(&[
            ("zero", &[0], 1),
            ("five", &[5], 120),
            ("ten", &[10], 3_628_800),
        ])
    }

    #[test]
    fn a_program_that_passes_every_test_passes() {
        let report = check(FACTORIAL, factorial_task());
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(report.level, VerificationLevel::Deterministic);
        assert_eq!(report.score.get(), 1.0);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn loops_branches_logic_and_printing_work() {
        let source = "
            fn f(n) {
                let total = 0; let i = 1;
                while i <= n {
                    if i % 2 == 0 || i == 3 { total = total + i; } else if i > 100 { total = 0; } else { print i; }
                    i = i + 1;
                }
                return total * -1 + !0 - (7 / 2) + (0 && 1);
            }";
        let mut t = task(&[("six", &[6], -(2 + 4 + 6 + 3) + 1 - 3)]);
        t.tests[0].expected_output = Some("1\n5\n".into());
        assert_eq!(check(source, t.clone()).status, VerificationStatus::Pass);
        t.tests[0].expected_output = Some("1\n".into());
        let report = check(source, t);
        assert_eq!(codes(&report), ["code.test_failed"]);
        assert!(report.findings[0].message.contains("output"));
    }

    #[test]
    fn syntax_errors_are_findings() {
        for source in [
            "",
            "fn f(n) { return n }",
            "fn f(n) { return n; ",
            "fn f(n) return n;",
            "fn f(n, n) { return 1; }",
            "fn f(n) { return 1; } fn f(n) { return 2; }",
            "fn f(n) { let = 1; }",
            "fn f(n) { return 99999999999999999999; }",
            "fn f(n) { return n $ 1; }",
            "fn f(n) { return 1; } extra",
            "fn f(n) { return \u{e9}; }",
            "fn let(n) { return 1; }",
            "fn g(n) { return n; }",
        ] {
            let report = check(source, task(&[("t", &[1], 1)]));
            assert_eq!(report.status, VerificationStatus::Fail, "{source:?}");
            assert_eq!(codes(&report), ["code.syntax"], "{source:?}");
            assert!(report.findings[0].hard);
        }
        let report = check("\n\nfn f(n) { return n }", task(&[("t", &[1], 1)]));
        assert!(report.findings[0].message.starts_with("line 3"));
    }

    #[test]
    fn each_failing_test_is_a_finding_with_its_id() {
        let report = check(
            "fn f(n) { return n + 1; }",
            task(&[("a", &[1], 2), ("b", &[2], 9), ("c", &[3], 0)]),
        );
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(codes(&report), ["code.test_failed", "code.test_failed"]);
        assert!(report.findings[0]
            .message
            .starts_with("test b: expected 9, got 3"));
        assert!(report.findings[1].message.starts_with("test c:"));
    }

    #[test]
    fn runtime_errors_are_test_failures_not_panics() {
        for (source, why) in [
            ("fn f(n) { return 1 / (n - n); }", "division by zero"),
            ("fn f(n) { return 1 % 0; }", "division by zero"),
            ("fn f(n) { return 9223372036854775807 + n; }", "overflow"),
            (
                "fn f(n) { return -9223372036854775807 - 1 - n; }",
                "overflow",
            ),
            (
                "fn f(n) { return -(0 - 9223372036854775807 - 1) * n; }",
                "overflow",
            ),
            ("fn f(n) { return x; }", "undeclared"),
            ("fn f(n) { x = 1; return x; }", "undeclared"),
            ("fn f(n) { return g(n); }", "undefined function"),
            ("fn f(n) { return f(); }", "arguments"),
            ("fn f(n) { n = 1; }", "without returning"),
        ] {
            let report = check(source, task(&[("t", &[1], 1)]));
            assert_eq!(codes(&report), ["code.test_failed"], "{source}");
            assert!(
                report.findings[0].message.contains(why),
                "{source}: {:?}",
                report.findings
            );
        }
    }

    #[test]
    fn infinite_loops_time_out_deterministically() {
        for source in [
            "fn f(n) { while 1 { } return 0; }",
            "fn f(n) { while 1 { n = n + 0; } return 0; }",
            "fn f(n) { let i = 0; while i >= 0 { i = i + 1; if i > 9223372036854775000 { i = 0; } } return 0; }",
            "fn f(n) { return 1 + g(n); } fn g(n) { return f(n); }",
        ] {
            let report = check(source, task(&[("spin", &[1], 0)]));
            assert_eq!(report.status, VerificationStatus::Fail, "{source}");
            assert!(report.findings.iter().all(|f| f.hard));
            assert!(
                codes(&report) == ["code.timeout"] || codes(&report) == ["code.resource"],
                "{source}: {:?}",
                report.findings
            );
        }
        let report = check(
            "fn f(n) { while 1 { } return 0; }",
            task(&[("spin", &[1], 0)]),
        );
        assert_eq!(codes(&report), ["code.timeout"]);
        assert!(report.findings[0].message.contains("test spin"));
        assert_eq!(
            report,
            check(
                "fn f(n) { while 1 { } return 0; }",
                task(&[("spin", &[1], 0)])
            )
        );
    }

    #[test]
    fn the_timeout_is_per_test_and_does_not_hide_other_tests() {
        let source = "fn f(n) { if n == 0 { while 1 { } } return n; }";
        let report = check(
            source,
            task(&[("loop", &[0], 0), ("ok", &[4], 4), ("bad", &[5], 6)]),
        );
        assert_eq!(codes(&report), ["code.timeout", "code.test_failed"]);
    }

    #[test]
    fn resource_bounds_are_findings() {
        // Unbounded recursion hits the call-depth bound before the step bound.
        let report = check("fn f(n) { return f(n + 1); }", task(&[("rec", &[0], 0)]));
        assert_eq!(codes(&report), ["code.resource"]);
        assert!(report.findings[0].message.contains("call depth"));
        // Output bound.
        let report = check(
            "fn f(n) { let i = 0; while i < 1000 { print 123456789; i = i + 1; } return 0; }",
            task(&[("out", &[0], 0)]),
        );
        assert_eq!(codes(&report), ["code.resource"]);
        assert!(report.findings[0].message.contains("output"));
        // Memory bound: more live variables than the bound, over several frames.
        let lets: String = (0..MAX_LOCALS + 1)
            .map(|i| format!("let v{i} = {i};"))
            .collect();
        let source = format!("fn f(n) {{ {lets} return 0; }}");
        let report = check(&source, task(&[("mem", &[0], 0)]));
        assert!(source.len() < MAX_SOURCE_BYTES && source.matches("let").count() > MAX_LOCALS);
        assert_eq!(codes(&report), ["code.resource"]);
        assert!(report.findings[0].message.contains("memory"));
        for finding in &report.findings {
            assert!(finding.hard);
        }
    }

    #[test]
    fn oversize_and_deeply_nested_sources_are_findings() {
        let big = format!("fn f(n) {{ return {}0; }}", "0 + ".repeat(MAX_SOURCE_BYTES));
        assert_eq!(
            codes(&check(&big, task(&[("t", &[0], 0)]))),
            ["code.oversize"]
        );
        // Few bytes per token but many tokens.
        let many = format!("fn f(n) {{ return {}0; }}", "1+".repeat(MAX_TOKENS));
        let report = check(&many, task(&[("t", &[0], 0)]));
        assert!(many.len() < MAX_SOURCE_BYTES);
        assert_eq!(codes(&report), ["code.syntax"]);
        let nested = format!(
            "fn f(n) {{ return {}1{}; }}",
            "(".repeat(500),
            ")".repeat(500)
        );
        assert_eq!(
            codes(&check(&nested, task(&[("t", &[0], 1)]))),
            ["code.syntax"]
        );
        let blocks = format!(
            "fn f(n) {{ {} return 1; {} }}",
            "if 1 {".repeat(300),
            "}".repeat(300)
        );
        assert_eq!(
            codes(&check(&blocks, task(&[("t", &[0], 1)]))),
            ["code.syntax"]
        );
        let unary = format!("fn f(n) {{ return {}1; }}", "-".repeat(500));
        assert_eq!(
            codes(&check(&unary, task(&[("t", &[0], 1)]))),
            ["code.syntax"]
        );
        // A long left-leaning chain stays within the height bound or is refused.
        let chain = format!("fn f(n) {{ return {}0; }}", "n + ".repeat(300));
        assert_eq!(
            codes(&check(&chain, task(&[("t", &[0], 0)]))),
            ["code.syntax"]
        );
        let elses = format!(
            "fn f(n) {{ {} return 1; }}",
            "if 0 { } else ".repeat(100) + "{ }"
        );
        assert_eq!(
            codes(&check(&elses, task(&[("t", &[0], 1)]))),
            ["code.syntax"]
        );
    }

    #[test]
    fn maximum_depth_programs_run_on_a_small_stack() {
        let source = "fn f(n) { if n == 0 { return 0; } return 1 + f(n - 1); }";
        let t = task(&[(
            "deep",
            &[MAX_CALL_DEPTH as i64 - 1],
            MAX_CALL_DEPTH as i64 - 1,
        )]);
        let handle = std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(move || {
                let nest = format!(
                    "{}n{}",
                    "(".repeat(MAX_NESTING - 2),
                    ")".repeat(MAX_NESTING - 2)
                );
                let source2 = format!(
                    "fn f(n) {{ if n == 0 {{ return 0; }} return 1 + f(n - 1) + 0 * {nest}; }}"
                );
                (
                    check(source, t.clone()).status,
                    check(&source2, t).status,
                    codes(&check(
                        source,
                        task(&[("over", &[MAX_CALL_DEPTH as i64 + 5], 0)]),
                    ))
                    .len(),
                )
            })
            .unwrap();
        let (a, b, n) = handle.join().unwrap();
        assert_eq!(a, VerificationStatus::Pass);
        assert_eq!(b, VerificationStatus::Pass);
        assert_eq!(n, 1);
    }

    #[test]
    fn failing_test_details_are_hidden_unless_the_test_is_visible() {
        let hidden = |t: CodeTest| CodeTask {
            entry: "f".into(),
            tests: vec![t],
        };
        // Default: only the test id, for a wrong value, a wrong output and a runtime error.
        for source in [
            "fn f(n) { return 41 + n; }",
            "fn f(n) { return 1 / (n - n); }",
        ] {
            let t = CodeTest::new("secret", &[1], 123_456);
            assert_eq!(t.visibility, TestVisibility::Hidden);
            let report = check(source, hidden(t));
            assert_eq!(codes(&report), ["code.test_failed"]);
            assert_eq!(report.findings[0].message, "test secret: failed");
        }
        let mut t = CodeTest::new("out", &[1], 1);
        t.expected_output = Some("secret-output\n".into());
        let report = check("fn f(n) { print 7; return n; }", hidden(t.clone()));
        assert_eq!(report.findings[0].message, "test out: failed");
        t.visibility = TestVisibility::Visible;
        let report = check("fn f(n) { print 7; return n; }", hidden(t));
        assert!(report.findings[0]
            .message
            .contains("printed output differs"));
        // Visible: the values and the runtime error.
        let report = check(
            "fn f(n) { return 41 + n; }",
            hidden(CodeTest::new("v", &[1], 7).visible()),
        );
        assert_eq!(report.findings[0].message, "test v: expected 7, got 42");
        let report = check(
            "fn f(n) { return 1 / (n - n); }",
            hidden(CodeTest::new("v", &[1], 7).visible()),
        );
        assert!(report.findings[0].message.contains("division by zero"));
        // Timeout and resource findings carry no test data in either mode.
        let report = check(
            "fn f(n) { while 1 { } return 0; }",
            hidden(CodeTest::new("h", &[1], 7)),
        );
        assert_eq!(codes(&report), ["code.timeout"]);
    }

    #[test]
    fn call_frames_cost_fuel_and_names_are_slots() {
        // Many distinct names in a callee: every call pays for its frame, so a loop of calls
        // runs out of fuel instead of doing unbounded unmetered work.
        let lets: String = (0..200).map(|i| format!("let v{i} = {i};")).collect();
        let source = format!(
            "fn g(n) {{ {lets} return n; }} fn f(n) {{ let i = 0; while i < 100000 {{ let t = g(i); i = i + 1; }} return 0; }}"
        );
        assert!(source.len() < MAX_SOURCE_BYTES);
        let report = check(&source, task(&[("calls", &[0], 0)]));
        assert_eq!(codes(&report), ["code.timeout"]);
        // Shadowing by re-`let` overwrites the same slot, and frames do not leak between calls.
        let source = "fn g(n) { let a = n; let a = a + 1; return a; }
                      fn f(n) { let i = 0; let s = 0; while i < 50 { s = s + g(i); i = i + 1; } return s; }";
        let expected: i64 = (0..50).map(|i| i + 1).sum();
        assert_eq!(
            check(source, task(&[("many", &[0], expected)])).status,
            VerificationStatus::Pass
        );
    }

    /// The deepest program the individual bounds admit: call depth 20, 12 nested blocks and a
    /// 13-deep parenthesised sum in every frame. The total native recursion bound stops it with
    /// `code.resource` instead of overflowing a 512 KiB stack (debug build, large frames); a
    /// program just inside the bound runs on the same stack.
    #[test]
    fn maximal_nesting_and_recursion_fit_in_a_small_stack() {
        let program = |blocks: usize| {
            let open = "if 1 { ".repeat(blocks);
            let close = " }".repeat(blocks);
            let sum = format!("{}f(n - 1){}", "1 + (".repeat(blocks), ")".repeat(blocks));
            format!(
                "fn f(n) {{ if n == 0 {{ return 0; }} {open} return {sum}; {close} return 0; }}"
            )
        };
        let maximal = program(12);
        assert!(maximal.len() < MAX_SOURCE_BYTES);
        let modest = program(2);
        let depth = MAX_CALL_DEPTH as i64;
        let handle = std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(move || {
                (
                    // 19 calls deep, 12-deep nesting each: over the native bound.
                    check(&maximal, task(&[("deepest", &[depth - 1], 0)])),
                    // Shallow nesting at the full call depth: inside every bound.
                    check(&modest, task(&[("modest", &[depth - 1], (depth - 1) * 2)])),
                    // Beyond the call-depth bound.
                    check(&modest, task(&[("over", &[depth + 10], 0)])),
                )
            })
            .unwrap();
        let (maximal, modest, over) = handle.join().expect("no stack overflow");
        assert_eq!(codes(&maximal), ["code.resource"], "{:?}", maximal.findings);
        assert!(maximal.findings[0].message.contains("native recursion"));
        assert_eq!(
            modest.status,
            VerificationStatus::Pass,
            "{:?}",
            modest.findings
        );
        assert_eq!(codes(&over), ["code.resource"]);
        // Parser recursion limits (within the 8 KiB source bound) on the same small stack.
        let handle = std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(|| {
                let deep = format!(
                    "fn f(n) {{ return {}1{}; }}",
                    "(".repeat(1500),
                    ")".repeat(1500)
                );
                let blocks = format!("fn f(n) {{ {} }}", "while 1 {".repeat(900));
                let unary = format!("fn f(n) {{ return {}1; }}", "!".repeat(3000));
                [deep, blocks, unary]
                    .iter()
                    .map(|s| {
                        assert!(s.len() <= MAX_SOURCE_BYTES);
                        codes(&check(s, task(&[("t", &[0], 0)]))).join(",")
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap();
        assert_eq!(handle.join().unwrap(), ["code.syntax"; 3]);
    }

    #[test]
    fn a_bad_task_is_unknown() {
        let many: Vec<_> = (0..=MAX_TESTS)
            .map(|i| CodeTest::new(&format!("t{i}"), &[], 0))
            .collect();
        for bad in [
            CodeTask {
                entry: "f".into(),
                tests: vec![],
            },
            CodeTask {
                entry: "f".into(),
                tests: many,
            },
            CodeTask {
                entry: "f".into(),
                tests: vec![CodeTest::new("", &[], 0)],
            },
            CodeTask {
                entry: "f".into(),
                tests: vec![CodeTest::new("a", &[], 0), CodeTest::new("a", &[], 0)],
            },
            CodeTask {
                entry: String::new(),
                tests: vec![CodeTest::new("a", &[], 0)],
            },
            CodeTask {
                entry: "f".into(),
                tests: vec![CodeTest::new("a", &[0; 9], 0)],
            },
        ] {
            let report = check("fn f() { return 0; }", bad);
            assert_eq!(report.status, VerificationStatus::Unknown);
            assert_eq!(codes(&report), ["code.task"]);
            assert!(!report.findings[0].hard);
        }
    }

    #[test]
    fn verification_is_deterministic_and_named() {
        let verifier = CodeTestVerifier::new(factorial_task());
        for source in [
            FACTORIAL,
            "fn f(n) { while 1 {} return 1; }",
            "garbage",
            "fn f(n) { return n; }",
        ] {
            let source = source.to_owned();
            assert_eq!(verifier.verify(&source), verifier.verify(&source));
        }
        assert_eq!(NamedVerifier::<String>::name(&verifier), "code-test");
        assert_eq!(verifier.task().entry, "f");
    }
}
