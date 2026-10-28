"""Session memory: conversation text for follow-ups, never evidence.

History is kept only for the current selection (split, engine, cutoff). Changing the selection or
the model/corpus version clears it, and earlier answers are given to the LLM as background text
that is explicitly not evidence; every assessment is recomputed from tool calls.
"""
from typing import List, Optional, Tuple

from Agent.schemas.contracts import RequestContext


class Session:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.key: Optional[Tuple] = None
        self.turns: List[Tuple[str, str]] = []

    @staticmethod
    def key_for(ctx: RequestContext) -> Tuple:
        return (ctx.dataset_split, ctx.engine_id, ctx.cutoff_cycle, ctx.model_version, ctx.corpus_version)

    def note_context(self, ctx: RequestContext) -> bool:
        """Clear history if the selection or versions changed. Returns True when it was cleared."""
        k = self.key_for(ctx)
        changed = self.key is not None and k != self.key
        if changed:
            self.turns = []
        self.key = k
        return changed

    def add(self, ctx: RequestContext, question: str, answer_text: str) -> None:
        self.note_context(ctx)
        self.turns.append((question, answer_text))
        self.turns = self.turns[-self.max_turns:]

    def history_text(self, ctx: RequestContext) -> str:
        if self.key != self.key_for(ctx) or not self.turns:
            return ""
        lines = [f"Earlier question: {q}\nEarlier reply (background only, NOT evidence): {a[:400]}" for q, a in self.turns[-3:]]
        return "\n\n".join(lines)

    def reset(self) -> None:
        self.key, self.turns = None, []
