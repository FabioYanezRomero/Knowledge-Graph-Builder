"""A malformed chunk costs that chunk, never the document.

The repair layer in ``providers.ollama`` handles the malformed shapes we have
seen. These tests cover the ones we have NOT: they feed the real langextract
resolver inputs it genuinely raises on, and assert the damage stops at the
chunk. Each case first asserts the plain ``Resolver`` really does raise, so the
test cannot quietly go vacuous if langextract gets more tolerant.
"""
import pytest

from langextract import data, extraction, resolver as lx_resolver
from langextract.core import format_handler as fh

from kgb.clients.chunk_safety import _BaseResolver, ChunkSafeResolver, install


def _resolver(cls):
    """Build `cls` the way langextract.extraction builds its resolver.

    Note `_BaseResolver`, not `lx_resolver.Resolver`: importing kgb.clients has
    already swapped the module attribute, so the "unguarded" half of these tests
    has to reach for the pristine class captured before install().
    """
    handler, rest = fh.FormatHandler.from_resolver_params(
        resolver_params={"require_extractions_key": False},
        base_format_type=data.FormatType.JSON,
        base_use_fences=True,
        base_attribute_suffix=data.ATTRIBUTE_SUFFIX,
        base_use_wrapper=True,
        base_wrapper_key=data.EXTRACTIONS_KEY,
    )
    return cls(format_handler=handler, **rest)


def _fenced(body):
    return f"```json\n{body}\n```"


GOOD = _fenced('{"extractions": [{"Triple": "A does B", "Triple_attributes": {"head": "A"}}]}')

# Both captured from real runs: 12b emitted a bare string among the extractions,
# and a nested extraction_text is the shape that killed 15 pathology reports.
FATAL = [
    pytest.param(
        _fenced('{"extractions": [{"Triple": "A does B", "Triple_attributes": {"head": "A"}},'
                ' "a bare string"]}'),
        id="non-mapping item",
    ),
    pytest.param(
        _fenced('{"extractions": [{"Triple": {"nested": 1}, "Triple_attributes": {"head": "A"}}]}'),
        id="nested extraction text",
    ),
]


@pytest.mark.parametrize("payload", FATAL)
def test_fatal_chunk_yields_nothing_instead_of_raising(payload):
    with pytest.raises(Exception):
        _resolver(_BaseResolver).resolve(payload)      # unguarded: kills the document
    assert _resolver(ChunkSafeResolver).resolve(payload) == []  # guarded: kills the chunk


def test_good_chunk_unaffected():
    safe = _resolver(ChunkSafeResolver).resolve(GOOD)
    assert [e.extraction_text for e in safe] == ["A does B"]
    assert safe[0].attributes == {"head": "A"}


def test_alignment_failure_keeps_the_extractions():
    # 26b hit "Source tokens and extraction tokens cannot be empty". Dropping the
    # extractions would be pure loss: kgb re-anchors every span to the document
    # itself, so langextract's offsets are not what we keep them for.
    safe = _resolver(ChunkSafeResolver)
    extractions = safe.resolve(GOOD)
    with pytest.raises(Exception):
        list(_resolver(_BaseResolver).align(extractions, "", 0, 0))
    kept = safe.align(extractions, "", 0, 0)
    assert [e.extraction_text for e in kept] == ["A does B"]


def test_install_is_what_lx_extract_actually_builds():
    # extraction.py does `resolver.Resolver(**params)` — a module-attribute lookup
    # at call time. If that ever becomes a direct import, this test fails and the
    # guarantee is silently gone.
    install()
    assert extraction.resolver.Resolver is ChunkSafeResolver


def test_installed_by_importing_the_clients_package():
    import kgb.clients  # noqa: F401
    assert lx_resolver.Resolver is ChunkSafeResolver
