"""Chunk-size sweep: does consolidation now absorb the fragmentation that small
chunks cause, and does recall still collapse on large chunks?

Answers the open question left in issue #2 by re-running its experiment against
the current pipeline. Reports, per chunk size, the same columns the issue used
(triples / entities / near-duplicate pairs / time) plus the after-consolidation
counts, so "how much did it improve" is a table, not an impression.

Usage:
    python scripts/chunk_size_sweep.py \
        --input data/legal/dxc_complaint.txt \
        --domain legal --client ollama --model gemma4:e4b \
        --chunk-sizes 1500,3000,6000,10000,16000,0

    (0 = no chunking: the whole document in one call.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kgb.builder.consolidation import entity_resolution_strategy
from kgb.builder.consolidation.layers.fuzzy import fuzzy_candidates
from kgb.builder.extraction import extract_triples
from kgb.clients import ClientConfig, ClientFactory
from kgb.domains import get_domain


def _entities(triples, standalone: list[str]) -> set[str]:
    """Every distinct node in the graph, standalone entities included."""
    nodes = {t.head for t in triples} | {t.tail for t in triples if t.tail}
    return nodes | set(standalone)


def _measure(triples, standalone: list[str]) -> dict:
    ents = _entities(triples, standalone)
    return {
        "triples": len(triples),
        "entities": len(ents),
        # Near-duplicates are the fragmentation signal: the same real entity
        # surfacing under two spellings because two chunks never saw each other.
        "near_dupes": len(fuzzy_candidates(sorted(ents))),
    }


def _num_ctx_for(chunk_chars: int) -> int:
    """Context window that actually fits a chunk of this size.

    Critical for this experiment: a fixed num_ctx silently truncates large
    chunks, which would look exactly like "the model loses recall on large
    chunks" while really being a config artefact. Issue #2 listed this as one of
    the candidate explanations, so the sweep scales the window instead of
    holding it fixed, and reports it per row.

    ~3.5 chars/token for English legal prose, plus room for prompt + examples
    (~2k tokens) and the generated triples (~2k tokens).
    """
    needed = int(chunk_chars / 3.5) + 4096
    ctx = 8192
    while ctx < needed and ctx < 131072:
        ctx *= 2
    return ctx


def sweep(text: str, domain_name: str, client_type: str, model: str,
          chunk_sizes: list[int], timeout: int, prompt: str | None = None) -> list[dict]:
    domain = get_domain(domain_name)
    rows = []
    for size in chunk_sizes:
        # 0 means "one call for the whole document" — langextract chunks on
        # max_char_buffer, so a buffer >= len(text) disables chunking.
        buffer = len(text) if size == 0 else size
        num_ctx = _num_ctx_for(buffer)

        def _client(ctx: int, char_buffer: int):
            return ClientFactory.create(ClientConfig(
                client_type=client_type,
                model_id=model,
                temperature=0.0,
                max_char_buffer=char_buffer,
                timeout=timeout,
                show_progress=False,
                # Local-backend practices: serialize the batch and disable the
                # reasoning phase, or the server wedges (see local-model-extraction).
                max_workers=1 if client_type in ("ollama", "lmstudio") else None,
                think=False if client_type == "ollama" else None,
                options={"num_ctx": ctx} if client_type == "ollama" else None,
            ))

        client = _client(num_ctx, buffer)
        # Consolidation sends ONE prompt holding every entity and its edge
        # context, so it is far larger than any extraction chunk — on this
        # document, 39K chars against a 1500-char chunk. Sizing its window from
        # the chunk truncates the response to nothing; size it from the document.
        consolidate_client = _client(_num_ctx_for(len(text)), len(text))

        t0 = time.perf_counter()
        triples, standalone = extract_triples(client, domain, text, prompt_override=prompt)
        extract_s = time.perf_counter() - t0

        before = _measure(triples, standalone)

        t0 = time.perf_counter()
        resolved, meta = entity_resolution_strategy(consolidate_client, domain, text, triples)
        consolidate_s = time.perf_counter() - t0

        after = _measure(resolved, standalone)
        rows.append({
            "chunk": "full" if size == 0 else str(size),
            "chunks": max(1, -(-len(text) // buffer)),
            "num_ctx": num_ctx,
            "before": before,
            "after": after,
            "extract_s": round(extract_s),
            "consolidate_s": round(consolidate_s),
            "status": meta.get("status", "ok"),
        })
        print(f"  {rows[-1]['chunk']:>6}: {before['triples']:>4} triples, "
              f"{before['entities']:>4} entities, {before['near_dupes']:>3} near-dupes "
              f"-> after consolidation {after['entities']:>4} / {after['near_dupes']:>3}",
              flush=True)
    return rows


def to_markdown(rows: list[dict], doc: str, chars: int, model: str) -> str:
    out = [
        f"### {doc} ({chars:,} chars) — {model}",
        "",
        "| Chunk size | Chunks | num_ctx | Triples | Density /1K | Entities | Near-dupes | Entities after | Near-dupes after | Fragmentation resolved | Extract | Consolidate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        b, a = r["before"], r["after"]
        # The point of the experiment: of the fragmentation small chunks
        # introduce, how much does consolidation now remove?
        resolved_pct = "n/a" if not b["near_dupes"] else \
            f"{round(100 * (b['near_dupes'] - a['near_dupes']) / b['near_dupes'])}%"
        out.append(
            f"| {r['chunk']} | {r['chunks']} | {r['num_ctx']} | {b['triples']} | "
            f"{round(1000 * b['triples'] / chars, 1)} | {b['entities']} | "
            f"{b['near_dupes']} | {a['entities']} | {a['near_dupes']} | {resolved_pct} | "
            f"{r['extract_s']}s | {r['consolidate_s']}s |"
        )
    out.append("")
    out.append("*Density /1K* (triples per 1,000 source chars) is the recall signal; "
               "*Fragmentation resolved* is the consolidation signal. `num_ctx` is "
               "scaled to the chunk so a truncated context cannot masquerade as lost recall.")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="A .txt document")
    p.add_argument("--domain", default="legal")
    p.add_argument("--client", default="ollama")
    p.add_argument("--model", required=True)
    p.add_argument("--chunk-sizes", default="1500,3000,6000,10000,16000,0",
                   help="Comma-separated char buffers; 0 = whole document")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--prompt-file", type=Path,
                   help="Extraction prompt to use instead of the domain's "
                        "(must keep the {{record_json}} placeholder). For A/B "
                        "testing prompt wording with everything else held fixed.")
    p.add_argument("--output", type=Path, help="Append the markdown table here")
    args = p.parse_args()

    text = args.input.read_text()
    sizes = [int(s) for s in args.chunk_sizes.split(",")]
    prompt = args.prompt_file.read_text() if args.prompt_file else None
    label = f" [prompt: {args.prompt_file.name}]" if args.prompt_file else ""
    print(f"{args.input.name}: {len(text):,} chars, {len(sizes)} runs on {args.model}{label}")

    rows = sweep(text, args.domain, args.client, args.model, sizes, args.timeout, prompt)
    table = to_markdown(rows, args.input.stem + label, len(text), args.model)

    print("\n" + table)
    if args.output:
        with args.output.open("a") as fh:
            fh.write("\n" + table + "\n")
        print(f"\nAppended to {args.output}")


if __name__ == "__main__":
    main()
