#!/usr/bin/env python3
"""Synthesize pairwise preference data for MedJudge by injecting controlled errors.

Implements the five candidate-generation strategies described in the MedJudge
paper. For every reference QA example we keep the correct response as
``reference`` and synthesize an inferior/alternative ``candidate`` whose origin is
recorded in ``candidate_type``:

    paper strategy                          candidate_type        driver
    -------------------------------------   -------------------   ------------------
    Random Pairing (RP)                     random                programmatic
    Sentence-level Perturbation (SLP)       omission              programmatic
    Semantic Similarity Pairing (SSP)       semantic_similarity   embedding retrieval
    Model-generated Alternatives (MGA)      model_generated       LLM sampling (GPT-4o)
    Structured Content Modification (SCM)   kg_contradiction      LLM rewrite (GPT-4o)

Mechanisms follow the paper's main text:
  * SSP retrieves, for each reference r_i, the most semantically similar response
    r_j (j != i) from another instance:  c_i = argmax_{j!=i} Sim(r_i, r_j).
  * MGA samples a plausible-but-diverse alternative from a medical QA model
    conditioned on the input:  c_i ~ P_theta(y | x_i).  Here P_theta is GPT-4o.

NOTE: In the *released* dataset these two strategies are stored under the labels
``subtle_error`` (= SSP) and ``paraphrase_error`` (= MGA). This script uses the
mechanism-accurate names above; pass ``--legacy-labels`` to emit the old labels.

Dependencies: ``openai`` (MGA/SCM) and ``sentence-transformers`` (SSP). The
programmatic strategies (random, omission) need neither.

Examples
--------
    # everything, GPT-4o + MiniLM retriever
    export OPENAI_API_KEY=...
    python scripts/synthesize_pairwise.py --input openended_train.json \
        --outdir Pairwise_Dataset --types all --model gpt-4o --merge

    # only the no-API strategies
    python scripts/synthesize_pairwise.py --input openended_train.json \
        --outdir Pairwise_Dataset --types random omission semantic_similarity
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

# candidate_type -> (paper strategy, legacy/released label)
STRATEGIES: dict[str, tuple[str, str]] = {
    "random":              ("Random Pairing (RP)",                  "random"),
    "omission":            ("Sentence-level Perturbation (SLP)",    "omission"),
    "semantic_similarity": ("Semantic Similarity Pairing (SSP)",    "subtle_error"),
    "model_generated":     ("Model-generated Alternatives (MGA)",   "paraphrase_error"),
    "kg_contradiction":    ("Structured Content Modification (SCM)", "kg_contradiction"),
}

# ----------------------------------------------------------------------------
# text helpers
# ----------------------------------------------------------------------------
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def strip_think(text: str) -> str:
    text = text or ""
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    return re.sub(r"</?think>", "", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_reference(record: dict[str, Any]) -> str:
    ref = record.get("reference") or record.get("output")
    if ref:
        return ref
    solution = (record.get("solution") or "").strip()
    answer = (record.get("answer") or "").strip()
    return f"<think>{solution}</think> The answer is {answer}."


# ----------------------------------------------------------------------------
# programmatic generators
# ----------------------------------------------------------------------------
def gen_random(records: list[dict[str, Any]], idx: int, rng: random.Random) -> tuple[str, Any]:
    """RP: borrow the reference of a different, unrelated instance."""
    if len(records) < 2:
        raise ValueError("Random pairing needs at least two records.")
    j = idx
    while j == idx:
        j = rng.randrange(len(records))
    return build_reference(records[j]), records[j].get("id", j)


def gen_omission(record: dict[str, Any], rng: random.Random,
                 keep_min: float = 0.4, keep_max: float = 0.7) -> tuple[str, Any]:
    """SLP: drop a subset of sentences so key findings/conclusion are omitted."""
    body = strip_think(build_reference(record))
    sents = split_sentences(body)
    if len(sents) <= 1:
        return body[: max(1, len(body) // 2)].rstrip(), record.get("id")
    keep_n = max(1, int(round(len(sents) * rng.uniform(keep_min, keep_max))))
    return " ".join(sents[:keep_n]), record.get("id")


# ----------------------------------------------------------------------------
# SSP: semantic-similarity retrieval  (c_i = argmax_{j!=i} Sim(r_i, r_j))
# ----------------------------------------------------------------------------
def compute_ssp_neighbors(records: list[dict[str, Any]], model_name: str,
                          chunk: int = 512) -> list[int]:
    """Return, for each index i, the index j!=i of the most similar response."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "SSP needs sentence-transformers + numpy: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    texts = [strip_think(build_reference(r)) for r in records]
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    n = len(emb)
    neighbors = [0] * n
    for s in range(0, n, chunk):
        block = emb[s:s + chunk] @ emb.T          # cosine (rows are normalized)
        for r in range(block.shape[0]):
            i = s + r
            block[r, i] = -1.0                    # exclude self
            neighbors[i] = int(block[r].argmax())
    return neighbors


# ----------------------------------------------------------------------------
# LLM (GPT-4o) helpers for MGA and SCM
# ----------------------------------------------------------------------------
def _image_data_url(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def make_caller(model: str, temperature: float, system_prompt: str,
                max_retries: int = 5) -> Callable[..., str]:
    """Return ``call(prompt, image_path=None) -> text`` backed by an OpenAI model."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install openai to use the LLM strategies.") from exc

    client = OpenAI()  # OPENAI_API_KEY / OPENAI_BASE_URL from env

    def call(prompt: str, image_path: str | None = None) -> str:
        content: Any = prompt
        url = _image_data_url(image_path)
        if url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        last = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last}")

    return call


# MGA: P_theta(y | x_i) — answer the question like a medical QA model.
MGA_SYSTEM = (
    "You are a medical visual question answering model. Given a medical image "
    "(when provided) and a question, answer it directly using your own clinical "
    "reasoning. Output only the answer."
)
MGA_PROMPT = """\
Question: {question}{choices}

Give a brief clinical rationale and then state your final answer on a new line as
"The answer is <answer>." Answer the question yourself; do not refer to any other
text."""

# SCM: modify one structured medical attribute to contradict known knowledge.
SCM_SYSTEM = (
    "You are a medical data-augmentation assistant. You rewrite a correct answer "
    "into a flawed alternative for training a medical judge. Output ONLY the "
    "rewritten answer text, with no preamble and no markup."
)
SCM_PROMPT = """\
Question: {question}

Correct answer:
{reference}

Produce an answer that closely mirrors the correct answer's structure and wording
but alters one structured medical attribute so the statement CONTRADICTS
established medical knowledge — e.g. negate a true finding, reverse a cause/effect
or anatomical relationship, or assign a finding to the wrong condition. Keep the
surface form similar so the contradiction is the only change. Output only the
rewritten answer."""


def gen_model_generated(record: dict[str, Any], call: Callable[..., str],
                        with_image: bool) -> tuple[str, Any]:
    choices = record.get("choices") or []
    choices_str = f"\nOptions: {', '.join(map(str, choices))}" if choices else ""
    prompt = MGA_PROMPT.format(question=(record.get("question") or "").strip(),
                               choices=choices_str)
    img = record.get("image") if with_image else None
    return call(prompt, img).strip(), record.get("id")


def gen_kg_contradiction(record: dict[str, Any], call: Callable[..., str]) -> tuple[str, Any]:
    prompt = SCM_PROMPT.format(question=(record.get("question") or "").strip(),
                               reference=strip_think(build_reference(record)))
    return call(prompt).strip(), record.get("id")


# ----------------------------------------------------------------------------
# record assembly / IO
# ----------------------------------------------------------------------------
PASSTHROUGH = ("question", "choices", "answer", "image", "answer_type",
               "image_organ", "solution", "dataset_source", "question_type", "output")


def make_record(src: dict[str, Any], candidate: str, ctype: str, source_id: Any,
                legacy: bool) -> dict[str, Any]:
    rec = {k: src.get(k) for k in PASSTHROUGH}
    rec["reference"] = build_reference(src)
    rec["candidate"] = candidate
    rec["candidate_source_id"] = source_id
    rec["candidate_type"] = STRATEGIES[ctype][1] if legacy else ctype
    return rec


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else list(data.values())
    for i, r in enumerate(records):
        r.setdefault("id", i)
    return records


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--types", nargs="+", default=["all"],
                    choices=["all", *STRATEGIES.keys()])
    ap.add_argument("--per-type", type=int, default=0,
                    help="Cap candidates per type (0 = every record).")
    ap.add_argument("--model", default="gpt-4o", help="GPT model for MGA/SCM.")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                    help="sentence-transformers model for SSP "
                         "(e.g. pritamdeka/S-PubMedBert-MS-MARCO for biomedical).")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--with-image", action="store_true",
                    help="Attach the image to MGA prompts when the file exists.")
    ap.add_argument("--legacy-labels", action="store_true",
                    help="Emit released labels (subtle_error/paraphrase_error).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = load_records(args.input)
    print(f"Loaded {len(records)} reference records from {args.input}")

    types = list(STRATEGIES) if "all" in args.types else args.types

    # set up the two LLM callers (lazy) and the SSP index only if needed
    mga_call = scm_call = None
    if "model_generated" in types:
        mga_call = make_caller(args.model, args.temperature, MGA_SYSTEM)
    if "kg_contradiction" in types:
        scm_call = make_caller(args.model, args.temperature, SCM_SYSTEM)
    ssp_neighbors: list[int] | None = None
    if "semantic_similarity" in types:
        print(f"Building SSP retriever with '{args.embed_model}' ...")
        ssp_neighbors = compute_ssp_neighbors(records, args.embed_model)

    args.outdir.mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, Any]] = []

    for ctype in types:
        strategy = STRATEGIES[ctype][0]
        pool = records if not args.per_type else records[: args.per_type]
        out: list[dict[str, Any]] = []
        for idx, rec in enumerate(pool):
            try:
                if ctype == "random":
                    cand, src_id = gen_random(records, idx, rng)
                elif ctype == "omission":
                    cand, src_id = gen_omission(rec, rng)
                elif ctype == "semantic_similarity":
                    j = ssp_neighbors[idx]                       # type: ignore[index]
                    cand, src_id = build_reference(records[j]), records[j].get("id", j)
                elif ctype == "model_generated":
                    cand, src_id = gen_model_generated(rec, mga_call, args.with_image)  # type: ignore[arg-type]
                else:  # kg_contradiction
                    cand, src_id = gen_kg_contradiction(rec, scm_call)  # type: ignore[arg-type]
            except Exception as exc:
                print(f"  ! {ctype} record {idx} skipped: {exc}")
                continue
            if cand and cand.strip():
                out.append(make_record(rec, cand, ctype, src_id, args.legacy_labels))
            if (idx + 1) % 200 == 0:
                print(f"  {ctype}: {idx + 1}/{len(pool)}")

        dest = args.outdir / f"{ctype}_pairwise.json"
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        merged.extend(out)
        print(f"[{ctype}] {strategy}: wrote {len(out)} -> {dest}")

    if args.merge:
        rng.shuffle(merged)
        dest = args.outdir / "all_pairwise.json"
        dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[merged] wrote {len(merged)} -> {dest}")


if __name__ == "__main__":
    main()
