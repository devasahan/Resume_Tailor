"""Loads models.yaml and answers "which models can I use right now?"."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import config
from .base import Usage


@dataclass
class ModelSpec:
    id: str
    label: str
    provider: str  # anthropic | openai_compat | fake
    model: str
    key_env: str = ""
    base_url: str = ""
    price_in: float = 0.0  # USD per 1M input tokens
    price_out: float = 0.0  # USD per 1M output tokens
    data_region: str = "UNKNOWN"
    verified: bool = False
    settings: dict = field(default_factory=dict)
    notes: str = ""

    def api_key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""

    def has_key(self) -> bool:
        return self.provider == "fake" or bool(self.api_key())

    def cost(self, usage: Usage) -> float:
        return usage.input_tokens * self.price_in / 1e6 + usage.output_tokens * self.price_out / 1e6

    @property
    def max_output_tokens(self) -> int:
        return int(self.settings.get("max_output_tokens", 8000))


@dataclass
class ModelStatus:
    available: bool
    reason: str = ""


class Registry:
    def __init__(self, specs: list[ModelSpec]):
        self._specs = {s.id: s for s in specs}

    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        path = path or config.models_file()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs = [ModelSpec(**entry) for entry in raw.get("models", [])]
        ids = [s.id for s in specs]
        if len(ids) != len(set(ids)):
            raise ValueError("models.yaml has duplicate model ids")
        return cls(specs)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise KeyError(f"Unknown model '{model_id}'. Known: {', '.join(self._specs)}") from None

    def all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def status(self, spec: ModelSpec, allowed_regions: list[str]) -> ModelStatus:
        if spec.provider != "fake" and spec.data_region not in allowed_regions:
            return ModelStatus(False, f"region {spec.data_region} is not allowed (Settings > Allowed regions)")
        if not spec.has_key():
            return ModelStatus(False, f"no API key: set {spec.key_env}")
        return ModelStatus(True)

    def available(self, allowed_regions: list[str]) -> list[ModelSpec]:
        return [s for s in self.all() if self.status(s, allowed_regions).available]

    def pick_default(self, priority: list[str], allowed_regions: list[str]) -> str | None:
        """First available model following `priority`, else any available one."""
        avail = {s.id for s in self.available(allowed_regions)}
        for model_id in priority:
            if model_id in avail:
                return model_id
        return next(iter(avail), None)
