"""Type-relations are domain data (schema.json), not hardcoded in the builder.

validate_entity_types uses the domain's type_relations to detect typing
triples ("X is_type Y") and validate head/tail against allowed entity types.
The default preserves prior behavior; a domain can override for its own
conventions/language.
"""

from kgb.domains import get_domain, Triple
from kgb.domains.models import DEFAULT_TYPE_RELATIONS, DomainSchema
from kgb.builder.validation import collect_schema_constraints, validate_entity_types


def test_schema_default_type_relations():
    assert DomainSchema().type_relations == DEFAULT_TYPE_RELATIONS


def test_default_relation_detected_as_typing(tmp_path):
    domain_dir = tmp_path / "d"
    (domain_dir / "extraction").mkdir(parents=True)
    (domain_dir / "extraction" / "prompt_constrained.md").write_text("x")
    (domain_dir / "schema.json").write_text(
        '{"entity_types": ["Drug"], "relation_types": ["is_type"]}'
    )
    domain = get_domain(str(domain_dir), extraction_mode="constrained")
    constraints = collect_schema_constraints(domain)

    # "instance_of" is in the default set -> treated as a typing relation
    triple = Triple(head="Aspirin", relation="instance_of", tail="Drug")
    valid, _ = validate_entity_types(triple, {}, constraints)
    assert valid is True  # tail "Drug" matches an allowed entity type


def test_domain_can_override_type_relations(tmp_path):
    domain_dir = tmp_path / "es"
    (domain_dir / "extraction").mkdir(parents=True)
    (domain_dir / "extraction" / "prompt_constrained.md").write_text("x")
    # Spanish domain declares its own typing relation; default English set not used.
    (domain_dir / "schema.json").write_text(
        '{"entity_types": ["Farmaco"], "type_relations": ["es_tipo_de"]}'
    )
    domain = get_domain(str(domain_dir), extraction_mode="constrained")
    constraints = collect_schema_constraints(domain)

    assert "es_tipo_de" in constraints.normalized_type_relations
    assert "instance_of" not in constraints.normalized_type_relations  # override, not merge

    triple = Triple(head="Aspirina", relation="es_tipo_de", tail="Farmaco")
    valid, _ = validate_entity_types(triple, {}, constraints)
    assert valid is True
