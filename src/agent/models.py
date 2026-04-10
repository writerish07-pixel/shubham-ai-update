from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    call_id: str
    user_turns: int = 0
    agent_turns: int = 0
    interrupted: bool = False
    history: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.user_turns += 1
        self.history.append({"role": "user", "text": text})

    def add_agent(self, text: str) -> None:
        self.agent_turns += 1
        self.history.append({"role": "assistant", "text": text})

    @property
    def talk_ratio(self) -> tuple[float, float]:
        total = max(self.user_turns + self.agent_turns, 1)
        return self.user_turns / total, self.agent_turns / total
