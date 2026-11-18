#!/usr/bin/env python3
"""
Take a small, balanced sample of Meta's CRAG benchmark (Task 1) for the tutor.

CRAG (https://github.com/facebookresearch/CRAG, CC BY-NC 4.0) gives every
question 5 web pages (full HTML) that may or may not contain the answer. The
full Task 1 file is 705 MB of bzip2, so this streams it and stops as soon as
the sample is full; it never downloads the whole file.

The sample spreads questions evenly over CRAG's 8 question types (simple,
comparison, multi-hop, false premise, ...) and, within each type, over its 5
domains (finance, sports, music, movie, open).

  python3 tutor_eval/crag_sample.py --n 20 --out crag/sample.jsonl

Standard library only. The data is not ours to redistribute: keep it out of git.
"""

import argparse
import bz2
import itertools
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DATA_URL = ("https://media.githubusercontent.com/media/facebookresearch/CRAG/main/"
            "data/crag_task_1_and_2_dev_v4.jsonl.bz2")
QUESTION_TYPES = ("simple", "simple_w_condition", "comparison", "aggregation", "set",
                  "multi-hop", "post-processing", "false_premise")
DOMAINS = ("finance", "sports", "music", "movie", "open")


def stream_examples(url: str, max_examples: int):
    """Yield CRAG examples (dicts) from the bzip2 JSONL at *url*, decompressing as it downloads."""
    decomp = bz2.BZ2Decompressor()
    buf = b""
    seen = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
        while seen < max_examples:
            block = resp.read(1 << 20)
            if not block:
                break
            buf += decomp.decompress(block)
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line.strip():
                    seen += 1
                    yield json.loads(line)
                    if seen >= max_examples:
                        return


def usable(example) -> bool:
    """Has a question, a gold answer, and at least one page with HTML."""
    pages = example.get("search_results") or []
    return bool(example.get("query") and str(example.get("answer", "")).strip()
                and any((p.get("page_result") or "").strip() for p in pages))


def balanced_pick(pool, n: int):
    """Round-robin over question types, and over domains within each type."""
    by_type = defaultdict(lambda: defaultdict(list))
    for ex in pool:
        by_type[ex.get("question_type")][ex.get("domain")].append(ex)
    order = [t for t in QUESTION_TYPES if t in by_type] + sorted(t for t in by_type if t not in QUESTION_TYPES)
    per_type = {}
    for i, qtype in enumerate(order):
        by_domain = by_type[qtype]
        domains = sorted(by_domain, key=lambda d: (d not in DOMAINS, d))
        # Start each type on a different domain, so any first-N slice of the
        # sample mixes domains as well as types.
        domains = domains[i % len(domains):] + domains[:i % len(domains)]
        queues = [by_domain[d] for d in domains]
        per_type[qtype] = [ex for group in itertools.zip_longest(*queues) for ex in group if ex]
    picked = []
    for rnd in itertools.count():
        added = False
        for qtype in order:
            if len(picked) >= n:
                return picked
            if rnd < len(per_type[qtype]):
                picked.append(per_type[qtype][rnd])
                added = True
        if not added:
            return picked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20, help="questions in the sample")
    p.add_argument("--scan", type=int, default=400,
                   help="how many CRAG examples to read before picking (more = better balance, bigger download)")
    p.add_argument("--url", default=DATA_URL)
    p.add_argument("--out", default="crag/sample.jsonl")
    args = p.parse_args(argv)

    pool = []
    print(f"Streaming up to {args.scan} CRAG examples…", flush=True)
    for i, ex in enumerate(stream_examples(args.url, args.scan), start=1):
        if usable(ex):
            pool.append(ex)
        if i % 50 == 0:
            print(f"  read {i}", flush=True)
    if not pool:
        sys.exit("No usable examples were read.")
    sample = balanced_pick(pool, args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in sample:
            f.write(json.dumps(ex) + "\n")
    html_chars = sum(len(pg.get("page_result") or "") for ex in sample for pg in ex["search_results"])
    print(f"Wrote {len(sample)} questions to {out} ({html_chars / 1e6:.1f} MB of HTML)")
    print("  by type:  ", dict(Counter(ex["question_type"] for ex in sample)))
    print("  by domain:", dict(Counter(ex["domain"] for ex in sample)))


if __name__ == "__main__":
    main()
