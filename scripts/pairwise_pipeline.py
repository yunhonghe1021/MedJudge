#!/usr/bin/env python3
"""Quality-filter, de-duplicate, and split synthesized pairwise data.

Turns raw synthesized pairwise candidates (see ``synthesize_pairwise.py``) into
clean train/test splits, reproducing the data-curation stage of MedJudge:

    1. Quality filtering   - drop malformed / degenerate pairs (minimal loss).
    2. Exact de-duplication- collapse identical (question, reference, candidate).
    3. 7:3 train/test split.
    4. Leakage control     - guarantee no response leaks across the split.

Leakage key
-----------
The paper's text says train instances sharing "an identical query or response
pair" with the test set are removed. The *released* dataset, however, keeps 286
overlapping questions while having ZERO overlapping references/response-pairs, so
leakage was actually controlled at the **response** level. This script therefore
defaults to ``--leakage-key reference`` (reproducing the released invariant);
pass ``query`` or ``response_pair`` to change it.

Two split modes:
  * ``random_then_prune`` (default): random 7:3 split, then drop train records
    whose leakage key collides with the test set (the paper's described method).
  * ``group_by_reference``: assign whole reference groups to one split, so no
    record is ever pruned (leak-free by construction).

Input may be a single merged JSON (list or {id: record}) or a directory of
``*_pairwise.json`` files.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def norm(s: Any) -> str:
    return " ".join(str(s or "").split()).lower().strip()


def strip_think(s: Any) -> str:
    s = str(s or "")
    m = THINK_RE.search(s)
    return (m.group(1) if m else re.sub(r"</?think>", "", s)).strip()


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
def load_any(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        records: list[dict[str, Any]] = []
        for f in sorted(path.glob("*_pairwise.json")):
            if f.name == "all_pairwise.json":
                continue
            records.extend(_load_one(f))
        return records
    return _load_one(path)


def _load_one(f: Path) -> list[dict[str, Any]]:
    data = json.loads(f.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else list(data.values())


# ----------------------------------------------------------------------------
# 1. quality filter
# ----------------------------------------------------------------------------
def quality_filter(records: list[dict[str, Any]], min_chars: int) -> tuple[list, Counter]:
    kept, reasons = [], Counter()
    for r in records:
        q, ref, cand = r.get("question"), r.get("reference"), r.get("candidate")
        if not norm(q):
            reasons["missing_question"] += 1; continue
        if not norm(ref):
            reasons["missing_reference"] += 1; continue
        if not norm(cand):
            reasons["empty_candidate"] += 1; continue
        if len(str(cand).strip()) < min_chars:
            reasons["candidate_too_short"] += 1; continue
        # degenerate: candidate identical to the reference -> no preference signal
        if strip_think(cand).lower() == strip_think(ref).lower():
            reasons["candidate_equals_reference"] += 1; continue
        kept.append(r)
    return kept, reasons


# ----------------------------------------------------------------------------
# 2. exact de-duplication
# ----------------------------------------------------------------------------
def dedup(records: list[dict[str, Any]]) -> tuple[list, int]:
    seen, kept = set(), []
    for r in records:
        key = (norm(r.get("question")), norm(r.get("reference")), norm(r.get("candidate")))
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(records) - len(kept)


# ----------------------------------------------------------------------------
# leakage keys
# ----------------------------------------------------------------------------
def leak_keys(record: dict[str, Any], which: list[str]) -> list[Any]:
    keys = []
    if "query" in which:
        keys.append(("q", norm(record.get("question"))))
    if "reference" in which:
        keys.append(("r", norm(record.get("reference"))))
    if "response_pair" in which:
        keys.append(("p", norm(record.get("reference")), norm(record.get("candidate"))))
    return keys


# ----------------------------------------------------------------------------
# 3+4. split
# ----------------------------------------------------------------------------
def split_random_then_prune(records, ratio, which, rng) -> tuple[list, list, int]:
    recs = list(records)
    rng.shuffle(recs)
    cut = int(round(len(recs) * ratio))
    train, test = recs[:cut], recs[cut:]
    test_keys = set()
    for r in test:
        test_keys.update(leak_keys(r, which))
    pruned = [r for r in train if not (set(leak_keys(r, which)) & test_keys)]
    return pruned, test, len(train) - len(pruned)


def split_group_by_reference(records, ratio, rng) -> tuple[list, list, int]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[norm(r.get("reference"))].append(r)
    keys = list(groups)
    rng.shuffle(keys)
    target = len(records) * ratio
    train, test, n = [], [], 0
    for k in keys:
        if n < target:
            train.extend(groups[k]); n += len(groups[k])
        else:
            test.extend(groups[k])
    return train, test, 0


# ----------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------
def overlap_report(train, test) -> dict[str, int]:
    def s(recs, fn):
        return set(fn(r) for r in recs)
    q = s(train, lambda r: norm(r.get("question"))) & s(test, lambda r: norm(r.get("question")))
    ref = s(train, lambda r: norm(r.get("reference"))) & s(test, lambda r: norm(r.get("reference")))
    pair = (s(train, lambda r: (norm(r.get("reference")), norm(r.get("candidate"))))
            & s(test, lambda r: (norm(r.get("reference")), norm(r.get("candidate")))))
    return {"question_overlap": len(q), "reference_overlap": len(ref), "response_pair_overlap": len(pair)}


def type_counts(records) -> dict[str, int]:
    return dict(Counter(str(r.get("candidate_type")) for r in records))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="Merged pairwise JSON, or a directory of *_pairwise.json.")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--split-ratio", type=float, default=0.7, help="Train fraction (paper: 0.7).")
    # group_by_reference is the DEFAULT: it reproduces the released splits'
    # invariants (0 response leakage, queries may overlap) and sizes. The
    # paper's literal "random then prune" prose discards ~75% of train and does
    # NOT reproduce the released sizes, so it is offered only for faithfulness.
    ap.add_argument("--split-mode", choices=["group_by_reference", "random_then_prune"],
                    default="group_by_reference")
    ap.add_argument("--leakage-key", default="reference",
                    help="Comma list of {reference,response_pair,query}. "
                         "Default 'reference' reproduces the released splits.")
    ap.add_argument("--min-candidate-chars", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    which = [k.strip() for k in args.leakage_key.split(",") if k.strip()]
    rng = random.Random(args.seed)

    records = load_any(args.input)
    print(f"Loaded {len(records)} raw pairwise records")

    records, reasons = quality_filter(records, args.min_candidate_chars)
    print(f"[1] quality filter -> {len(records)} kept; dropped {dict(reasons)}")

    records, n_dup = dedup(records)
    print(f"[2] dedup -> {len(records)} kept; removed {n_dup} exact duplicates")

    if args.split_mode == "group_by_reference":
        train, test, pruned = split_group_by_reference(records, args.split_ratio, rng)
    else:
        train, test, pruned = split_random_then_prune(records, args.split_ratio, which, rng)
    print(f"[3] split ({args.split_mode}, ratio={args.split_ratio}) -> "
          f"train {len(train)} / test {len(test)}; pruned {pruned} from train (keys={which})")

    rep = overlap_report(train, test)
    print(f"[4] leakage check -> {rep}")
    if rep["reference_overlap"] or rep["response_pair_overlap"]:
        print("    WARNING: response-level leakage remains; check --leakage-key.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "train_all_pairwise.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.outdir / "test_all_pairwise.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "raw_loaded": len(records) + n_dup,
        "after_filter_dedup": len(records),
        "train": len(train), "test": len(test),
        "train_test_ratio": round(len(train) / max(1, len(train) + len(test)), 4),
        "pruned_from_train": pruned,
        "filter_reasons": dict(reasons),
        "duplicates_removed": n_dup,
        "leakage": rep,
        "train_type_counts": type_counts(train),
        "test_type_counts": type_counts(test),
    }
    (args.outdir / "split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote train/test + split_stats.json to {args.outdir}")


if __name__ == "__main__":
    main()
