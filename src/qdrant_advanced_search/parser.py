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
class FilterClause:
    """A payload field filter clause.

    Attributes:
        field: Payload field name to filter on.
        values: List of values to match (strings or integers).
        exclude: If True, exclude matching documents (NOT IN). Default is IN.
    """

    field: str
    values: list[str | int]
    exclude: bool = False


@dataclass
class PrefetchClause:
    """Prefetch clause for a complex query.

    Attributes:
        kind: Either "sem" or "keywords".
        sem: Semantic expression, present when kind == "sem".
        keywords: Boolean expression, present when kind == "keywords".
        lim: Optional limit for semantic prefetch.
        filters: Optional list of payload field filters.
    """

    kind: str
    sem: SemExpr | None = None
    keywords: BoolNode | None = None
    lim: int | None = None
    filters: list[FilterClause] = field(default_factory=list)


@dataclass
class ReqClause:
    """Request clause for a complex query.

    Attributes:
        kind: Either "sem" or "keywords".
        sem: Semantic expression, present when kind == "sem".
        keywords: Boolean expression, present when kind == "keywords".
        lim: Optional result limit.
        filters: Optional list of payload field filters.
    """

    kind: str
    sem: SemExpr | None = None
    keywords: BoolNode | None = None
    lim: int | None = None
    filters: list[FilterClause] = field(default_factory=list)


@dataclass
class SimpleQuery:
    """A plain text search query (no c: prefix).

    Attributes:
        text: The raw search text.
        filters: Optional list of payload field filters.
    """

    text: str
    filters: list[FilterClause] = field(default_factory=list)


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
TT_LIM_COLON = "LIM_COLON"
TT_FILTER_COLON = "FILTER_COLON"
TT_AND = "AND"
TT_OR = "OR"
TT_NOT = "NOT"
TT_IN = "IN"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_LBRACKET = "LBRACKET"
TT_RBRACKET = "RBRACKET"
TT_COMMA = "COMMA"
TT_QUOTED = "QUOTED"
TT_NUMBER = "NUMBER"
TT_WORD = "WORD"

_TOKEN_PATTERNS: list[tuple[str, str]] = [
    (TT_C_COLON, r"c\s*:"),
    (TT_PRE_COLON, r"pre\s*:"),
    (TT_REQ_COLON, r"req\s*:"),
    (TT_SEM_COLON, r"sem\s*:"),
    (TT_KEYWORDS_COLON, r"keywords\s*:"),
    (TT_LIM_COLON, r"lim\s*:"),
    (TT_FILTER_COLON, r"filter\s*:"),
    (TT_AND, r"AND(?=[\s()\"\[])"),
    (TT_OR, r"OR(?=[\s()\"\[])"),
    (TT_NOT, r"NOT(?=[\s()\"\[])"),
    (TT_IN, r"IN(?=[\s\[])"),
    (TT_LPAREN, r"\("),
    (TT_RPAREN, r"\)"),
    (TT_LBRACKET, r"\["),
    (TT_RBRACKET, r"\]"),
    (TT_COMMA, r","),
    (TT_QUOTED, r'"[^"]*"'),
    (TT_NUMBER, r"\d+"),
    (TT_WORD, r'[^\s()\[\]",]+'),
]

_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_PATTERNS),
    re.IGNORECASE,
)

# Tokens that mark the boundary of a bool expression (stop parsing)
_BOOL_STOPS = {TT_LIM_COLON, TT_FILTER_COLON, TT_REQ_COLON}


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
            val = m.group(0)[1:-1]  # strip surrounding double quotes
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
# Boolean expression parser
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
        if isinstance(child, Literal):
            return Literal(value=child.value, negated=not child.negated)
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
# Filter clause parser
# ---------------------------------------------------------------------------


def _parse_filter_clause(stream: _Stream) -> FilterClause:
    """Parse a single filter: field IN [...] or filter: field NOT IN [...] clause.

    Args:
        stream: The token stream positioned at filter:.

    Returns:
        A FilterClause.

    Raises:
        ValueError: If the syntax is malformed.
    """
    stream.consume(TT_FILTER_COLON)

    # Field name — accept any non-structural token
    tok = stream.peek()
    if tok is None or tok.type in _BOOL_STOPS | {TT_REQ_COLON}:
        raise ValueError("Expected field name after filter:")
    field_name = stream.consume().value

    # IN or NOT IN
    nxt = stream.peek()
    if nxt and nxt.type == TT_NOT:
        stream.consume(TT_NOT)
        stream.consume(TT_IN)
        exclude = True
    elif nxt and nxt.type == TT_IN:
        stream.consume(TT_IN)
        exclude = False
    else:
        raise ValueError(f"Expected IN or NOT IN after field name '{field_name}', got {nxt!r}")

    # [ value, value, ... ]
    stream.consume(TT_LBRACKET)
    values: list[str | int] = []
    while True:
        tok = stream.peek()
        if tok is None or tok.type == TT_RBRACKET:
            break
        if tok.type == TT_QUOTED:
            values.append(stream.consume(TT_QUOTED).value)
        elif tok.type == TT_NUMBER:
            values.append(int(stream.consume(TT_NUMBER).value))
        elif tok.type == TT_WORD:
            values.append(stream.consume(TT_WORD).value)
        else:
            break
        # Optional comma separator
        if (nxt := stream.peek()) and nxt.type == TT_COMMA:
            stream.consume(TT_COMMA)

    stream.consume(TT_RBRACKET)
    return FilterClause(field=field_name, values=values, exclude=exclude)


def _parse_filter_clauses(stream: _Stream) -> list[FilterClause]:
    """Parse zero or more consecutive filter: clauses.

    Args:
        stream: The current token stream.

    Returns:
        List of FilterClause objects.
    """
    clauses: list[FilterClause] = []
    while not stream.at_end() and (nxt := stream.peek()) and nxt.type == TT_FILTER_COLON:
        clauses.append(_parse_filter_clause(stream))
    return clauses


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
    nxt = stream.peek()
    nxt2 = stream.peek(1)
    if nxt and nxt.type == TT_NOT and nxt2 and nxt2.type == TT_QUOTED:
        stream.consume(TT_NOT)
        negative = stream.consume(TT_QUOTED).value
    return SemExpr(positive=positive, negative=negative)


def _parse_modifiers(
    stream: _Stream,
    *,
    allow_lim: bool = True,
    stop_at_req: bool = True,
) -> tuple[int | None, list[FilterClause]]:
    """Parse optional lim: and filter: modifiers in any order.

    Args:
        stream: The current token stream.
        allow_lim: Whether lim: is valid in this context.
        stop_at_req: Whether to stop when req: is encountered.

    Returns:
        Tuple of (lim, filters).
    """
    lim: int | None = None
    filters: list[FilterClause] = []

    while not stream.at_end():
        nxt = stream.peek()
        if nxt is None:
            break
        if stop_at_req and nxt.type == TT_REQ_COLON:
            break
        if allow_lim and nxt.type == TT_LIM_COLON:
            stream.consume(TT_LIM_COLON)
            lim = int(stream.consume(TT_NUMBER).value)
        elif nxt.type == TT_FILTER_COLON:
            filters.append(_parse_filter_clause(stream))
        else:
            break

    return lim, filters


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

    if tok and tok.type == TT_SEM_COLON:
        sem = _parse_sem(stream)
        kind = "sem"
        lim, filters = _parse_modifiers(stream, allow_lim=True, stop_at_req=True)
        return PrefetchClause(kind=kind, sem=sem, lim=lim, filters=filters)

    if tok and tok.type == TT_KEYWORDS_COLON:
        stream.consume(TT_KEYWORDS_COLON)
        keywords = _parse_bool_expr(stream)
        # lim: not valid for keywords prefetch
        _, filters = _parse_modifiers(stream, allow_lim=False, stop_at_req=True)
        return PrefetchClause(kind="keywords", keywords=keywords, filters=filters)

    raise ValueError(f"Expected sem: or keywords: after pre:, got {tok!r}")


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

    if tok and tok.type == TT_SEM_COLON:
        sem = _parse_sem(stream)
        kind = "sem"
    elif tok and tok.type == TT_KEYWORDS_COLON:
        stream.consume(TT_KEYWORDS_COLON)
        keywords = _parse_bool_expr(stream)
        kind = "keywords"
    elif tok and tok.type == TT_QUOTED:
        # Bare quoted string treated as sem positive
        positive = stream.consume(TT_QUOTED).value
        sem = SemExpr(positive=positive)
        kind = "sem"
    else:
        raise ValueError(
            f"Expected sem:, keywords:, or quoted string after req:, got {tok!r}"
        )

    lim, filters = _parse_modifiers(stream, allow_lim=True, stop_at_req=False)

    if kind == "sem":
        return ReqClause(kind=kind, sem=sem, lim=lim, filters=filters)  # type: ignore[possibly-undefined]
    return ReqClause(kind=kind, keywords=keywords, lim=lim, filters=filters)  # type: ignore[possibly-undefined]


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
        # Simple query — extract any trailing filter: clauses
        filter_match = re.search(r"(?i)\bfilter:", stripped)
        if filter_match:
            query_text = stripped[: filter_match.start()].strip()
            tokens = _tokenize(stripped[filter_match.start():])
            stream = _Stream(tokens=tokens)
            filters = _parse_filter_clauses(stream)
            return SimpleQuery(text=query_text, filters=filters)
        return SimpleQuery(text=stripped)

    tokens = _tokenize(stripped)
    stream = _Stream(tokens=tokens)

    stream.consume(TT_C_COLON)

    prefetch: PrefetchClause | None = None
    if (nxt := stream.peek()) and nxt.type == TT_PRE_COLON:
        prefetch = _parse_prefetch(stream)

    if stream.at_end() or (stream.peek() and stream.peek().type != TT_REQ_COLON):  # type: ignore[union-attr]
        raise ValueError(
            "Une requête complexe (c:) doit contenir une clause req:. "
            "Exemple : c: pre: keywords: \"alcene\" req: sem: \"structure\""
        )

    req = _parse_req(stream)
    return ComplexQuery(prefetch=prefetch, req=req)
