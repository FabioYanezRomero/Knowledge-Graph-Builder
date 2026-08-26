"""One malformed chunk must never cost the whole document.

langextract's annotator calls ``resolver.resolve()`` and ``resolver.align()``
once per chunk with no error handling around either, so a single unparseable
response aborts the entire document — including every chunk that had already
succeeded. On a 60k-char document at chunk 1500 that is 40 good chunks thrown
away because the 41st came back malformed.

We do repair the known malformed shapes upstream (see
``providers.ollama._coerce_scalar_values``), but *known* is the weak word there:
every model we tested broke a different way — duplicate JSON keys, non-mapping
items, a double-encoded payload stuffed into an object key, the input field
echoed back instead of an extraction. Enumerating shapes is a treadmill that a
new model or a new document length restarts. Bounding the blast radius is not.

So this is the guarantee, and repair is only an optimisation on top of it:

* ``resolve`` raises  -> that chunk yields nothing; the document continues.
* ``align`` raises    -> the extractions are kept **unaligned**. kgb re-anchors
  every span to the document itself (``builder.extraction``, by exact find), so
  losing langextract's offsets costs nothing; dropping the extractions would.

Both paths print, because a dropped chunk is real data loss — just a much
smaller loss than the document.
"""

from __future__ import annotations

from langextract import resolver as _resolver

_BaseResolver = _resolver.Resolver


class ChunkSafeResolver(_BaseResolver):
    """A ``Resolver`` whose failures cost one chunk instead of the document."""

    def resolve(self, input_text, *args, **kwargs):
        try:
            return super().resolve(input_text, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 — the whole point is "whatever it is"
            print(f"  [langextract] chunk dropped, could not resolve "
                  f"({type(e).__name__}: {e}); rest of the document unaffected")
            return []

    def align(self, extractions, *args, **kwargs):
        try:
            # align() is a generator, so it only raises while being consumed:
            # materialise it inside the try or the except never fires.
            return list(super().align(extractions, *args, **kwargs))
        except Exception as e:  # noqa: BLE001
            print(f"  [langextract] chunk kept but not aligned "
                  f"({type(e).__name__}: {e}); kgb re-anchors spans itself")
            return list(extractions)


def install() -> None:
    """Make every ``lx.extract()`` call chunk-safe.

    ``extraction.py`` looks up ``resolver.Resolver`` at call time, so replacing
    the module attribute is enough and no langextract source is touched.
    """
    _resolver.Resolver = ChunkSafeResolver
