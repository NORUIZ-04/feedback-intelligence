from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def check_texts(cls, texts: list[str]) -> list[str]:
        cleaned = [" ".join(t.split()) for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("texts must not be empty or whitespace")
        if any(len(t) > 1000 for t in cleaned):
            raise ValueError("each text must be at most 1000 characters")
        return cleaned


class ClassifyItem(BaseModel):
    label: str
    score: float
    scores: dict[str, float]


class ClassifyResponse(BaseModel):
    results: list[ClassifyItem]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

    @field_validator("question")
    @classmethod
    def clean(cls, q: str) -> str:
        q = " ".join(q.split())
        if len(q) < 3:
            raise ValueError("question is too short")
        return q


class Source(BaseModel):
    n: int
    id: str
    doc: str
    rerank_score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    cited: list[int]
    sources: list[Source]
    timings: dict[str, float]