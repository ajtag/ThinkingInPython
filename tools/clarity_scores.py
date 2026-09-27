#!/usr/bin/env python3
"""Rate each prose paragraph's clarity with Jev, and flag the unclear ones.

Jev (TypeSafe AI's "System One" model) answers a rubric question with a
probability per level instead of prose, fast and cheaply enough to ask it
about every paragraph in the book. This tool asks one `Score` question
per paragraph, on the four-level `RUBRIC` below, and lists the paragraphs
whose expected score falls under `--threshold`. The point is triage: run
the expensive prose passes (`/straighten`, `/antecedents`, the
prose-clarity agent) on the paragraphs Jev flags instead of a whole
chapter.

Report only. It never edits a file and never joins a gate: its answers
come from a network model, so they are neither deterministic nor free,
the two properties every gate here has.

The SDK is deliberately not a project dependency. Run it with `uv run
--with typesafe-sdk`, which overlays the package for that one run and
leaves `pyproject.toml` and `uv.lock` alone:

    uv run --with typesafe-sdk python -m tools.clarity_scores 30
    uv run --with typesafe-sdk python -m tools.clarity_scores \\
        Solutions/30_*.md --threshold 2 --json

`TYPESAFE_API_KEY` must be set. `--dry-run` needs neither the key nor
the SDK: it counts the paragraphs and characters a run would send.

What Jev sees: the paragraph, the heading it sits under, the paragraph
before it, and the listing directly above it when there is one, so a
paragraph that explains a listing is not rated unclear for referring to
it. Link targets are stripped (their text stays). Only ordinary prose
paragraphs are rated; lists, tables, and block quotes are skipped, as
are paragraphs under `--min-words`.

Scores are cached in `build/clarity_scores.json`, keyed by a hash of
everything sent (the rubric included), so a rerun after editing a
chapter pays only for the paragraphs that changed.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.config import CHAPTERS_DIR, ROOT
from tools.markdown import HEADING, Document
from tools.prose import is_prose_line
from tools.repo import md_files

CACHE = ROOT / "build" / "clarity_scores.json"

# The listing above a paragraph is context, not the thing rated, so a
# long one is cut to its first lines to keep the token count down.
LISTING_LINES = 60

INSTRUCTIONS = (
    "Rate how easily an experienced programmer who is learning Python "
    "reads `paragraph` on the first pass. Text in backticks names real "
    "identifiers from the book's code; technical vocabulary is expected "
    "and does not make a paragraph unclear. `section`, `listing`, and "
    "`previous` are context: use them to resolve what the paragraph "
    "refers to, but rate only `paragraph`."
)

# Ordered from score 0 up. Level 1 lists the faults the repo's own
# clarity passes (/straighten, /antecedents, /literal) fix, so a flagged
# paragraph is one those passes have work to do on.
RUBRIC = (
    "Opaque: after two readings the reader still cannot say what the "
    "paragraph claims, or a key sentence reads two ways.",
    "Strained: the point comes through only after rereading a "
    "sentence: an actor buried or unnamed, a subject held open far from "
    "its verb, stacked negatives, a 'this' or 'it' with two candidates, "
    "a figure of speech standing in for a mechanism, or a step of "
    "reasoning left out.",
    "Clear: each sentence reads once; a phrase or two could be sharper.",
    "Plain: each sentence reads once, names its actor, and follows from "
    "the one before.",
)

_LINK = re.compile(r"\[([^\]]+)\]\([^)\s]+\)")


@dataclass(frozen=True)
class Paragraph:
    path: Path
    line: int
    """1-based line number of the paragraph's first line."""
    text: str
    section: str
    previous: str
    listing: str

    def state(self) -> dict[str, str]:
        """What Jev is shown, with empty context fields left out."""
        fields = {
            "section": self.section,
            "listing": self.listing,
            "previous": _unlink(self.previous),
            "paragraph": _unlink(self.text),
        }
        return {k: v for k, v in fields.items() if v}

    def key(self, model: str) -> str:
        blob = json.dumps(
            [model, INSTRUCTIONS, RUBRIC, self.state()], sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class Rating:
    score: float
    """Expected level on RUBRIC, 0 to 3; may fall between levels."""
    confidence: float
    input_tokens: int = 0


def _unlink(text: str) -> str:
    return _LINK.sub(r"\1", text)


def paragraphs(doc: Document) -> Iterator[Paragraph]:
    """Each ordinary prose paragraph in `doc`, with its context."""
    fenced = doc.in_fence()
    section = previous = listing = ""
    blocks = {b.open_at: b for b in doc.blocks}
    i, n = 0, len(doc.lines)
    while i < n:
        line = doc.lines[i]
        if block := blocks.get(i):
            listing = "\n".join(block.lines[:LISTING_LINES])
            i = Document.end_of(block)
            continue
        if fenced[i]:
            i += 1
            continue
        if m := HEADING.match(line):
            section, previous, listing = m.group(1), "", ""
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        if not is_prose_line(line):
            listing = ""  # a list or table now sits between them
            i += 1
            continue
        start = i
        while i < n and not fenced[i] and is_prose_line(doc.lines[i]):
            i += 1
        text = " ".join(s.strip() for s in doc.lines[start:i])
        yield Paragraph(doc.path, start + 1, text, section, previous,
                        listing)
        previous, listing = text, ""


def resolve(selector: str) -> list[Path]:
    """A file, a directory, or a chapter selector ("30", "Observer")."""
    path = Path(selector)
    if path.is_file() or path.is_dir():
        return md_files([path])
    if matches := sorted(CHAPTERS_DIR.glob(f"{selector}*.md")):
        return matches
    low = selector.lower()
    return sorted(
        p for p in CHAPTERS_DIR.glob("*.md") if low in p.stem.lower()
    )


def load_cache() -> dict[str, dict[str, Any]]:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except FileNotFoundError, json.JSONDecodeError:
        return {}


def save_cache(cache: dict[str, dict[str, Any]]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")


async def rate_all(
    todo: list[Paragraph], model: str | None, jobs: int
) -> dict[Paragraph, Rating]:
    """Ask Jev about every paragraph in `todo`, `jobs` at a time."""
    from typesafe_sdk import AsyncTypeSafeClient, Score

    question = Score(instructions=INSTRUCTIONS, criteria=list(RUBRIC))
    gate = asyncio.Semaphore(jobs)
    results: dict[Paragraph, Rating] = {}
    done = 0

    async def rate(client: AsyncTypeSafeClient, p: Paragraph) -> None:
        nonlocal done
        async with gate:
            response = await client.system_one(
                state=p.state(), questions={"clarity": question}
            )
        answer = response.scores["clarity"]
        results[p] = Rating(
            answer.score, answer.confidence,
            response.usage.input_tokens or 0,
        )
        done += 1
        print(f"\r  rated {done}/{len(todo)}", end="", file=sys.stderr)

    async with AsyncTypeSafeClient(model=model) as client:
        async with asyncio.TaskGroup() as tg:
            for p in todo:
                tg.create_task(rate(client, p))
    if todo:
        print(file=sys.stderr)
    return results


def excerpt(text: str, width: int = 72) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def report(
    rated: list[tuple[Paragraph, Rating]], threshold: float,
    show_all: bool, as_json: bool,
) -> int:
    flagged = 0
    for p, r in rated:
        low = r.score < threshold
        flagged += low
        if not (low or show_all):
            continue
        where = f"{p.path.relative_to(ROOT).as_posix()}:{p.line}"
        if as_json:
            row = {"path": where, "score": round(r.score, 3),
                   "confidence": round(r.confidence, 3),
                   "flagged": low, "text": p.text}
            print(json.dumps(row))
        else:
            mark = "!" if low else " "
            print(f"{mark} {r.score:4.2f} ({r.confidence:.2f})  {where}")
            print(f"      {excerpt(p.text)}")
    return flagged


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Flag unclear prose paragraphs, rated by Jev.",
    )
    ap.add_argument(
        "paths", nargs="*",
        help="files, directories, or chapter selectors like '30' or "
             "'Observer' (default: all of Chapters/)",
    )
    ap.add_argument(
        "--threshold", type=float, default=1.5,
        help="flag a paragraph scoring below this, on the rubric's "
             "0-3 scale (default: 1.5)",
    )
    ap.add_argument("--min-words", type=int, default=12,
                    help="skip shorter paragraphs (default: 12)")
    ap.add_argument("--model", help="Jev model (default: the SDK's)")
    ap.add_argument("-j", "--jobs", type=int, default=8,
                    help="requests in flight at once (default: 8)")
    ap.add_argument("--all", action="store_true",
                    help="print every paragraph, not only flagged ones")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object per line instead of text")
    ap.add_argument("--no-cache", action="store_true",
                    help="rate every paragraph again")
    ap.add_argument("--dry-run", action="store_true",
                    help="count what a run would send; call nothing")
    args = ap.parse_args()

    files: list[Path] = []
    for selector in args.paths or [str(CHAPTERS_DIR)]:
        if not (matched := resolve(selector)):
            print(f"no Markdown file matches: {selector}", file=sys.stderr)
            return 2
        files.extend(p.resolve() for p in matched)

    paras = [
        p for f in files for p in paragraphs(Document.parse(f))
        if len(p.text.split()) >= args.min_words
    ]
    model = args.model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "")
    if args.dry_run:
        chars = sum(len(json.dumps(p.state())) for p in paras)
        print(f"{len(paras)} paragraphs in {len(files)} files, "
              f"about {chars:,} characters of state")
        return 0
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 2
    try:
        import typesafe_sdk  # noqa: F401
    except ModuleNotFoundError:
        print("typesafe-sdk is not installed; run under "
              "`uv run --with typesafe-sdk`", file=sys.stderr)
        return 2

    cache = {} if args.no_cache else load_cache()
    ratings: dict[Paragraph, Rating] = {}
    for p in paras:
        if hit := cache.get(p.key(model)):
            ratings[p] = Rating(hit["score"], hit["confidence"])
    todo = [p for p in paras if p not in ratings]
    fresh = asyncio.run(rate_all(todo, args.model, args.jobs))
    ratings |= fresh
    cache.update({
        p.key(model): {k: v for k, v in asdict(r).items()
                       if k != "input_tokens"}
        for p, r in fresh.items()
    })
    save_cache(cache)

    rated = [(p, ratings[p]) for p in paras]
    flagged = report(rated, args.threshold, args.all, args.json)
    tokens = sum(r.input_tokens for r in fresh.values())
    print(
        f"{flagged} of {len(paras)} paragraphs below {args.threshold} "
        f"({len(fresh)} rated now, {len(paras) - len(fresh)} cached, "
        f"{tokens:,} input tokens)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
