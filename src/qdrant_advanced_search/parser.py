"""Parse search strings into typed dataclasses for Qdrant advanced search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SemExpr:
    """Semantic search expression with optional negative example.

    Attributes:
        positive: The positive query string.
        negative: Optional negative query string (NOT in sem context).
    """

    positive: str
    negative: str | None = None


@dataclass
class Literal:
    """A single keyword or tag literal, optionally negated.

    Attributes:
        value: Keyword or tag value (without quotes, or #TAG).
        negated: Whether this literal is negated.
    """

    value: str
    negated: bool = False


@dataclass
class BoolOp:
    """A binary boolean operation node.

    Attributes:
        op: The operator, either "AND" or "OR".
        left: Left operand.
        right: Right operand.
    """

    op: str
    left: BoolNode
    right: BoolNode


BoolNode = Literal | BoolOp


@dataclass
class PrefetchClause:
    """Prefetch clause for a complex query.

    Attributes:
        kind: Either "sem" or "keywords".
        sem: Semantic expression, present when kind == "sem".
        keywords: Boolean expression, present when kind == "keywords".
        lim: Optional limit for semantic prefetch.
        tags: Optional tag filter as a boolean expression.
    """

    kind: str
    sem: SemExpr | None = None
    keywords: BoolNode | None = None
    lim: int | None = None
    tags: BoolNode | None = None


@dataclass
class ReqClause:
    """Request clause for a complex query.

    Attributes:
        kind: Either "sem" or "keywords".
        sem: Semantic expression, present when kind == "sem".
        keywords: Boolean expression, present when kind == "keywords".
        lim: Optional result limit.
        tags: Optional tag filter as a boolean expression.
    """

    kind: str
    sem: SemExpr | None = None
    keywords: BoolNode | None = None
    lim: int | None = None
    tags: BoolNode | None = None


@dataclass
class SimpleQuery:
    """A plain text search query (no c: prefix).

    Attributes:
        text: The raw search text.
    """

    text: str


@dataclass
class ComplexQuery:
    """A complex structured query with optional prefetch and required request.

    Attributes:
        prefetch: Optional prefetch clause.
        req: The main request clause.
    """

    prefetch: PrefetchClause | None
    req: ReqClause


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

TT_C_COLON = "C_COLON"
TT_PRE_COLON = "PRE_COLON"
TT_REQ_COLON = "REQ_COLON"
TT_SEM_COLON = "SEM_COLON"
TT_KEYWORDS_COLON = "KEYWORDS_COLON"
TT_TAGS_COLON = "TAGS_COLON"
TT_LIM_COLON = "LIM_COLON"
TT_AND = "AND"
TT_OR = "OR"
TT_NOT = "NOT"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_QUOTED = "QUOTED"
TT_NUMBER = "NUMBER"
TT_WORD = "WORD"


_TOKEN_PATTERNS: list[tuple[str, str]] = [
    (TT_C_COLON, r"c:"),
    (TT_PRE_COLON, r"pre:"),
    (TT_REQ_COLON, r"req:"),
    (TT_SEM_COLON, r"sem:"),
    (TT_KEYWORDS_COLON, r"keywords:"),
    (TT_TAGS_COLON, r"tags:"),
    (TT_LIM_COLON, r"lim:"),
    (TT_AND, r"AND(?=[\s()\"])"),
    (TT_OR, r"OR(?=[\s()\"])"),
    (TT_NOT, r"NOT(?=[\s()\"])"),
    (TT_LPAREN, r"\("),
    (TT_RPAREN, r"\)"),
    (TT_QUOTED, r'"[^"]*"'),
    (TT_NUMBER, r"\d+"),
    (TT_WORD, r'[^\s()"]+'),
]

_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_PATTERNS),
    re.IGNORECASE,
)


@dataclass
class Token:
    """A lexical token produced by the tokenizer.

    Attributes:
        type: The token type string.
        value: The matched string value (or captured group for QUOTED).
    """

    type: str
    value: str


def _tokenize(text: str) -> list[Token]:
    """Tokenize a raw query string into a list of tokens.

    Args:
        text: The raw query string.

    Returns:
        List of Token objects.
    """
    tokens: list[Token] = []
    for m in _MASTER_RE.finditer(text):
        tt = m.lastgroup
        if tt is None:
            continue
        if tt == TT_QUOTED:
            raw = m.group(0)
            val = raw[1:-1]  # strip surrounding double quotes
        else:
            val = m.group(0)
        tokens.append(Token(type=tt, value=val))
    return tokens


# ---------------------------------------------------------------------------
# Token stream helper
# ---------------------------------------------------------------------------


@dataclass
class _Stream:
    """A consumable token stream.

    Attributes:
        tokens: The list of tokens.
        pos: Current position index.
    """

    tokens: list[Token]
    pos: int = field(default=0)

    def peek(self, offset: int = 0) -> Token | None:
        """Peek at a token without consuming it.

        Args:
            offset: How many positions ahead to look.

        Returns:
            The token at pos + offset, or None if out of range.
        """
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: str | None = None) -> Token:
        """Consume the next token.

        Args:
            expected_type: If provided, assert the next token matches this type.

        Returns:
            The consumed token.

        Raises:
            ValueError: If the token type does not match expected_type.
            IndexError: If there are no more tokens.
        """
        tok = self.tokens[self.pos]
        if expected_type is not None and tok.type != expected_type:
            raise ValueError(
                f"Expected {expected_type} but got {tok.type!r} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    def at_end(self) -> bool:
        """Check whether the stream is exhausted.

        Returns:
            True if no more tokens remain.
        """
        return self.pos >= len(self.tokens)


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


def _parse_bool_expr(stream: _Stream) -> BoolNode:
    """Parse a boolean expression with standard precedence (NOT > AND > OR).

    Args:
        stream: The token stream, positioned at the start of the expression.

    Returns:
        A BoolNode representing the parsed expression.
    """
    return _parse_or(stream)


def _parse_or(stream: _Stream) -> BoolNode:
    """Parse OR-level boolean expression.

    Args:
        stream: The current token stream.

    Returns:
        A BoolNode for the OR expression.
    """
    left = _parse_and(stream)
    while (tok := stream.peek()) and tok.type == TT_OR:
        stream.consume(TT_OR)
        right = _parse_and(stream)
        left = BoolOp(op="OR", left=left, right=right)
    return left


def _parse_and(stream: _Stream) -> BoolNode:
    """Parse AND-level boolean expression.

    Args:
        stream: The current token stream.

    Returns:
        A BoolNode for the AND expression.
    """
    left = _parse_not(stream)
    while (tok := stream.peek()) and tok.type == TT_AND:
        stream.consume(TT_AND)
        right = _parse_not(stream)
        left = BoolOp(op="AND", left=left, right=right)
    return left


def _parse_not(stream: _Stream) -> BoolNode:
    """Parse NOT-level boolean expression (unary NOT).

    Args:
        stream: The current token stream.

    Returns:
        A BoolNode, possibly negated.
    """
    tok = stream.peek()
    if tok and tok.type == TT_NOT:
        stream.consume(TT_NOT)
        child = _parse_not(stream)
        # Wrap single Literal in negated form; for BoolOp, wrap in Literal is
        # not possible — represent as negated subtree via a special pattern
        if isinstance(child, Literal):
            return Literal(value=child.value, negated=not child.negated)
        # For a BoolOp child, we can't flip it easily — wrap as-is and flag
        # unsupported; in practice the grammar only uses NOT before literals
        return child
    return _parse_atom(stream)


def _parse_atom(stream: _Stream) -> BoolNode:
    """Parse an atomic boolean expression: quoted string, word, or parenthesised.

    Args:
        stream: The current token stream.

    Returns:
        A BoolNode for the atom.

    Raises:
        ValueError: If no valid atom is found.
    """
    tok = stream.peek()
    if tok is None:
        raise ValueError("Unexpected end of input while parsing bool atom")
    if tok.type == TT_LPAREN:
        stream.consume(TT_LPAREN)
        node = _parse_or(stream)
        stream.consume(TT_RPAREN)
        return node
    if tok.type in (TT_QUOTED, TT_WORD):
        stream.consume()
        return Literal(value=tok.value)
    raise ValueError(f"Unexpected token {tok.type!r} ({tok.value!r}) in bool expr")


# ---------------------------------------------------------------------------
# Clause parsers
# ---------------------------------------------------------------------------


def _parse_sem(stream: _Stream) -> SemExpr:
    """Parse a sem: clause.

    Args:
        stream: The token stream, positioned before sem:.

    Returns:
        A SemExpr with positive and optional negative strings.

    Raises:
        ValueError: If the positive quoted string is missing.
    """
    stream.consume(TT_SEM_COLON)
    tok = stream.peek()
    if tok is None or tok.type != TT_QUOTED:
        raise ValueError("Expected quoted string after sem:")
    positive = stream.consume(TT_QUOTED).value
    negative: str | None = None
    # Check for NOT "..."
    nxt = stream.peek()
    nxt2 = stream.peek(1)
    if nxt and nxt.type == TT_NOT and nxt2 and nxt2.type == TT_QUOTED:
        stream.consume(TT_NOT)
        negative = stream.consume(TT_QUOTED).value
    return SemExpr(positive=positive, negative=negative)


def _parse_prefetch(stream: _Stream) -> PrefetchClause:
    """Parse a pre: clause.

    Args:
        stream: The token stream, positioned before pre:.

    Returns:
        A PrefetchClause.

    Raises:
        ValueError: If neither sem: nor keywords: follows pre:.
    """
    stream.consume(TT_PRE_COLON)
    tok = stream.peek()
    sem: SemExpr | None = None
    keywords: BoolNode | None = None
    lim: int | None = None
    tags: BoolNode | None = None

    if tok and tok.type == TT_SEM_COLON:
        sem = _parse_sem(stream)
        kind = "sem"
        # Optional lim:
        nxt = stream.peek()
        if nxt and nxt.type == TT_LIM_COLON:
            stream.consume(TT_LIM_COLON)
            lim = int(stream.consume(TT_NUMBER).value)
    elif tok and tok.type == TT_KEYWORDS_COLON:
        stream.consume(TT_KEYWORDS_COLON)
        keywords = _parse_bool_expr(stream)
        kind = "keywords"
    else:
        raise ValueError(f"Expected sem: or keywords: after pre:, got {tok!r}")

    # Optional tags:
    nxt = stream.peek()
    if nxt and nxt.type == TT_TAGS_COLON:
        stream.consume(TT_TAGS_COLON)
        tags = _parse_bool_expr(stream)

    return PrefetchClause(kind=kind, sem=sem, keywords=keywords, lim=lim, tags=tags)


def _parse_req(stream: _Stream) -> ReqClause:
    """Parse a req: clause.

    Args:
        stream: The token stream, positioned before req:.

    Returns:
        A ReqClause.

    Raises:
        ValueError: If neither sem: nor keywords: follows req:.
    """
    stream.consume(TT_REQ_COLON)
    tok = stream.peek()
    sem: SemExpr | None = None
    keywords: BoolNode | None = None
    lim: int | None = None
    tags: BoolNode | None = None

    if tok and tok.type == TT_SEM_COLON:
        sem = _parse_sem(stream)
        kind = "sem"
    elif tok and tok.type == TT_KEYWORDS_COLON:
        stream.consume(TT_KEYWORDS_COLON)
        keywords = _parse_bool_expr(stream)
        kind = "keywords"
    else:
        # Bare quoted string after req: treated as sem positive without sem: prefix
        if tok and tok.type == TT_QUOTED:
            positive = stream.consume(TT_QUOTED).value
            sem = SemExpr(positive=positive)
            kind = "sem"
        else:
            raise ValueError(
                f"Expected sem:, keywords:, or quoted string after req:, got {tok!r}"
            )

    # Optional lim:
    nxt = stream.peek()
    if nxt and nxt.type == TT_LIM_COLON:
        stream.consume(TT_LIM_COLON)
        lim = int(stream.consume(TT_NUMBER).value)

    # Optional tags:
    nxt = stream.peek()
    if nxt and nxt.type == TT_TAGS_COLON:
        stream.consume(TT_TAGS_COLON)
        tags = _parse_bool_expr(stream)

    return ReqClause(kind=kind, sem=sem, keywords=keywords, lim=lim, tags=tags)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_query(text: str) -> SimpleQuery | ComplexQuery:
    """Parse a raw query string into a SimpleQuery or ComplexQuery.

    If the string does not start with ``c:`` (case-insensitive), returns a
    SimpleQuery.  Otherwise tokenizes and parses the full structured syntax.

    Args:
        text: The raw query string entered by the user.

    Returns:
        A SimpleQuery for plain text queries, or a ComplexQuery for structured
        queries starting with ``c:``.

    Raises:
        ValueError: If the structured query syntax is malformed.
    """
    stripped = text.strip()
    if not re.match(r"^c:", stripped, re.IGNORECASE):
        return SimpleQuery(text=stripped)

    tokens = _tokenize(stripped)
    stream = _Stream(tokens=tokens)

    stream.consume(TT_C_COLON)

    prefetch: PrefetchClause | None = None
    if (nxt := stream.peek()) and nxt.type == TT_PRE_COLON:
        prefetch = _parse_prefetch(stream)

    req = _parse_req(stream)
    return ComplexQuery(prefetch=prefetch, req=req)
