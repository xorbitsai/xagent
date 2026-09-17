from enum import Enum

class FullTextOperator(str, Enum):
    AND = "AND"
    OR = "OR"

class Occur(str, Enum):
    SHOULD = "SHOULD"
    MUST = "MUST"
    MUST_NOT = "MUST_NOT"

class FullTextQuery:
    def to_json(self) -> str: ...

class MatchQuery(FullTextQuery):
    query: str
    column: str
    boost: float
    fuzziness: int
    max_expansions: int
    operator: FullTextOperator
    prefix_length: int
    def __init__(
        self,
        query: str,
        column: str,
        *,
        boost: float = ...,
        fuzziness: int = ...,
        max_expansions: int = ...,
        operator: FullTextOperator = ...,
        prefix_length: int = ...,
    ) -> None: ...

class BooleanQuery(FullTextQuery):
    queries: list[tuple[Occur, FullTextQuery]]
    def __init__(self, queries: list[tuple[Occur, FullTextQuery]]) -> None: ...
