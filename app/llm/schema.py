"""Turn a Pydantic model into a strict, self-contained JSON schema.

Strict means: no $ref/$defs, every property required, additionalProperties false. That is the
subset of JSON Schema that provider "structured output" features accept most reliably.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

_DROP = {"title", "default", "examples"}


def strict_json_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(x) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(defs[node["$ref"].split("/")[-1]])
        out: dict = {}
        for key, value in node.items():
            if key == "properties":  # property NAMES must survive even if they match _DROP
                out[key] = {name: resolve(sub) for name, sub in value.items()}
            elif key not in _DROP:
                out[key] = resolve(value)
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}).keys())
        return out

    return resolve(schema)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of model text, tolerating markdown fences and chatter around it."""
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("expected a JSON object", cleaned, 0)
    return value
