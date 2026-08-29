from knowledge.tools.validate_registry import validate


def test_knowledge_registry_is_consistent():
    assert validate() == []

