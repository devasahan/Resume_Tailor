import json

import pytest

from app.llm.schema import extract_json, strict_json_schema
from app.schemas import CoverLetterOut, Profile, TailorOutput, TailorRevision


def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


@pytest.mark.parametrize("model", [TailorOutput, TailorRevision, CoverLetterOut, Profile])
def test_strict_schema_is_self_contained_and_closed(model):
    schema = strict_json_schema(model)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    for node in walk(schema):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"].keys())


def test_no_property_is_named_after_a_json_schema_keyword():
    keywords = {"items", "type", "properties", "required", "enum", "format", "default", "title"}
    for node in walk(strict_json_schema(TailorOutput)):
        assert not keywords & set(node.get("properties", {})), node


def test_property_names_survive_even_if_they_look_like_keywords():
    schema = strict_json_schema(TailorOutput)
    assert "company_index" in schema["properties"]["sections"]["properties"]["experience"]["items"]["properties"]


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json here")
    with pytest.raises(json.JSONDecodeError):
        extract_json("[1, 2]")
