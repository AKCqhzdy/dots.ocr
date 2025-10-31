from dataclasses import dataclass
from typing import Literal, Optional

from openai.types import CompletionUsage
from pydantic import BaseModel


class TokenUsageItem(BaseModel):
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelIdentifier:
    id: str
    provider: str


class OcrTaskStats(BaseModel):
    status: Literal["pending", "running", "failed", "finished", "fallback", "timeout"]
    error_msg: Optional[str] = None
    attempt: int = 0
    # The execution time for the successful attempt
    task_execution_time: Optional[float] = None
    token_usage: dict[ModelIdentifier, CompletionUsage] = {}

    def add_token_usage(
        self, model_ident: ModelIdentifier, usage: Optional[CompletionUsage]
    ):
        if usage is None:
            return
        if model_ident not in self.token_usage:
            self.token_usage[model_ident] = usage
        else:
            total_usage = self.token_usage[model_ident]
            total_usage.completion_tokens += usage.completion_tokens
            total_usage.prompt_tokens += usage.prompt_tokens
            total_usage.total_tokens += usage.total_tokens
