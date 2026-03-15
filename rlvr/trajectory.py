"""
ATIF (Agent Trajectory Interchange Format) v1.6 — compatible with Harbor.

Records agent interactions as structured trajectories for RL training.
See: harbor/src/harbor/models/trajectories/
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


@dataclass
class ToolCall:
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass
class ObservationResult:
    source_call_id: Optional[str] = None
    content: Optional[str] = None


@dataclass
class Observation:
    results: list[ObservationResult] = field(default_factory=list)


@dataclass
class Metrics:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    extra: Optional[dict[str, Any]] = None


@dataclass
class Step:
    step_id: int
    source: str  # "system", "user", "agent"
    message: str
    timestamp: Optional[str] = None
    model_name: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    observation: Optional[Observation] = None
    metrics: Optional[Metrics] = None
    extra: Optional[dict[str, Any]] = None


@dataclass
class Agent:
    name: str
    version: str
    model_name: Optional[str] = None
    tool_definitions: Optional[list[dict[str, Any]]] = None
    extra: Optional[dict[str, Any]] = None


@dataclass
class FinalMetrics:
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None
    total_steps: Optional[int] = None
    extra: Optional[dict[str, Any]] = None


@dataclass
class Trajectory:
    schema_version: str = "ATIF-v1.6"
    session_id: str = field(default_factory=lambda: f"sc_{uuid4().hex[:12]}")
    agent: Agent = field(default_factory=lambda: Agent(name="SkillClaw", version="v3"))
    steps: list[Step] = field(default_factory=list)
    notes: Optional[str] = None
    final_metrics: Optional[FinalMetrics] = None
    extra: Optional[dict[str, Any]] = None


class TrajectoryRecorder:
    """Records agent interactions into ATIF format."""

    def __init__(self, model_name: str = None, tool_definitions: list = None):
        self.trajectory = Trajectory(
            agent=Agent(
                name="SkillClaw",
                version="v3",
                model_name=model_name,
                tool_definitions=tool_definitions,
            )
        )
        self._step_counter = 0

    def _next_id(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ")

    def add_system(self, message: str):
        self.trajectory.steps.append(Step(
            step_id=self._next_id(),
            source="system",
            message=message,
            timestamp=self._now(),
        ))

    def add_user(self, message: str):
        self.trajectory.steps.append(Step(
            step_id=self._next_id(),
            source="user",
            message=message,
            timestamp=self._now(),
        ))

    def add_agent(
        self,
        message: str,
        tool_calls: list[ToolCall] = None,
        model_name: str = None,
        reasoning_content: str = None,
        metrics: Metrics = None,
    ):
        self.trajectory.steps.append(Step(
            step_id=self._next_id(),
            source="agent",
            message=message,
            timestamp=self._now(),
            model_name=model_name,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            metrics=metrics,
        ))

    def add_observation(self, results: list[ObservationResult]):
        """Add observation as a system step (tool results)."""
        self.trajectory.steps.append(Step(
            step_id=self._next_id(),
            source="system",
            message="Tool execution results",
            timestamp=self._now(),
            observation=Observation(results=results),
        ))

    def finalize(self, extra: dict = None):
        """Compute final metrics and finalize trajectory."""
        total_steps = len(self.trajectory.steps)
        self.trajectory.final_metrics = FinalMetrics(
            total_steps=total_steps,
            extra=extra,
        )

    def to_dict(self) -> dict:
        """Export as dict, removing None values."""
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [clean(v) for v in obj]
            elif hasattr(obj, '__dataclass_fields__'):
                return clean(asdict(obj))
            return obj
        return clean(asdict(self.trajectory))

    def save(self, path: Path):
        """Save trajectory as JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        path.write_text(json.dumps(data, indent=2))
        return path
