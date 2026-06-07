#!/usr/bin/env python3
"""Evaluate MedJudge pairwise predictions with the paper's main metrics.

The script accepts JSONL predictions produced by the current pairwise inference
script. It also supports reward-model style outputs when each row has score_a
and score_b, prob_a and prob_b, or a scalar score/probability for label B.

For generative SFT/SFT+Rea outputs that only contain a hard A/B prediction, the
paper's AUC-ROC/AP are computed by treating the 0/1 hard prediction as the score:
predicted B -> 1.0, predicted A -> 0.0. RM outputs use continuous scores when
available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_gt(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise TypeError(f"Unsupported ground-truth type: {type(data)}")


def label_from_gt(item: dict[str, Any]) -> str:
    if "correct_answer" in item:
        return str(item["correct_answer"]).strip().upper()
    messages = item.get("messages") or item.get("conversations")
    if messages:
        last = messages[-1]
        return str(last.get("content") or last.get("value") or "").strip().upper()
    for key in ("label", "answer", "preferred"):
        value = item.get(key)
        if str(value).strip().upper() in {"A", "B"}:
            return str(value).strip().upper()
    return ""


def extract_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
        if tail:
            text = tail

    patterns = [
        r"(?:final\s+answer|answer|correct\s+answer)\s*(?:is|:)?\s*[\(\[]?\s*([AB])\b",
        r"\b([AB])\s*(?:is\s+better|is\s+more\s+accurate|is\s+correct)\b",
        r"^\s*[\(\[]?\s*([AB])\s*[\)\].:]?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        match = re.match(r"^[\(\[]?\s*([AB])\s*[\)\].:]?\s*$", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.match(r"^[\(\[]?\s*([AB])\b", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    upper = text.upper()
    has_a = re.search(r"\bA\b", upper) is not None
    has_b = re.search(r"\bB\b", upper) is not None
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return ""


def prediction_label(row: dict[str, Any]) -> str:
    for key in ("pred_label", "prediction", "label_pred", "pred"):
        value = row.get(key)
        if isinstance(value, str) and value.strip().upper() in {"A", "B"}:
            return value.strip().upper()

    if "score_a" in row and "score_b" in row:
        return "B" if float(row["score_b"]) >= float(row["score_a"]) else "A"
    if "prob_a" in row and "prob_b" in row:
        return "B" if float(row["prob_b"]) >= float(row["prob_a"]) else "A"

    return extract_answer(str(row.get("predict") or row.get("response") or row.get("output") or ""))


def score_for_b(row: dict[str, Any], pred: str) -> float:
    """Return the binary AUC-ROC score for class B.

    The paper reports AUC-ROC for the A-vs-B pairwise decision. When continuous
    reward/probability scores exist, use them. For SFT/SFT+Rea generation where
    only the final A/B decision is available, use the hard 0/1 prediction as the
    score.
    """
    if "prob_b" in row:
        return float(row["prob_b"])
    if "score_b" in row and "score_a" in row:
        sa = float(row["score_a"])
        sb = float(row["score_b"])
        diff = max(min(sb - sa, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-diff))
    for key in ("score", "prob", "probability", "confidence"):
        if key in row:
            value = float(row[key])
            if key == "confidence" and pred == "A":
                return 1.0 - value
            return value
    return 1.0 if pred == "B" else 0.0


def binary_auc(y_true: list[int], scores: list[float]) -> float | None:
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None

    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        positives = sum(label for _, label in pairs[i:j])
        rank_sum += positives * avg_rank
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(y_true: list[int], scores: list[float]) -> float | None:
    pos = sum(y_true)
    if pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if y_true[idx]:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / pos


def classification_metrics(labels: list[str], preds: list[str], scores_b: list[float]) -> dict[str, Any]:
    classes = ["A", "B"]
    total = len(labels)
    correct = sum(y == p for y, p in zip(labels, preds))
    per_class = {}
    f1s = []
    for cls in classes:
        tp = sum(y == cls and p == cls for y, p in zip(labels, preds))
        fp = sum(y != cls and p == cls for y, p in zip(labels, preds))
        fn = sum(y == cls and p != cls for y, p in zip(labels, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(y == cls for y in labels)}
        f1s.append(f1)

    y_binary = [1 if y == "B" else 0 for y in labels]
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "auc_roc": binary_auc(y_binary, scores_b),
        "average_precision": average_precision(y_binary, scores_b),
        "per_class": per_class,
    }


def strip_think(text: str) -> str:
    text = text or ""
    text = re.sub(r"</?think>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_umls_pipeline(model_name: str, threshold: float):
    try:
        import spacy
        import scispacy.linking  # noqa: F401 registers the spaCy factory
    except Exception as exc:
        raise RuntimeError(
            "UCO requires spacy and scispacy. Install them in the evaluation environment first."
        ) from exc

    nlp = spacy.load(model_name)
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": False,
            "linker_name": "umls",
            "threshold": threshold,
            "max_entities_per_mention": 1,
        },
    )
    return nlp


def compute_uco(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    preds: list[str],
    model_name: str,
    threshold: float,
    batch_size: int,
    text_mode: str,
) -> dict[str, Any]:
    pairs = []
    skipped = 0
    for gt, pred_row, pred_label in zip(ground_truth, predictions, preds):
        if text_mode == "explanation":
            reference = gt.get("explanation") or gt.get("rationale") or gt.get("solution")
            selected = pred_row.get("predict") or pred_row.get("response") or pred_row.get("output")
        elif text_mode == "selected_response":
            label = label_from_gt(gt)
            if label not in {"A", "B"} or pred_label not in {"A", "B"}:
                skipped += 1
                continue
            reference = gt.get(label)
            selected = gt.get(pred_label)
        else:
            raise ValueError(f"Unsupported UCO text mode: {text_mode}")

        if not reference or not selected:
            skipped += 1
            continue
        pairs.append((strip_think(str(reference)), strip_think(str(selected))))

    if not pairs:
        return {
            "available": False,
            "reason": "Ground truth does not contain A/B response text fields required for UCO.",
        }

    nlp = load_umls_pipeline(model_name, threshold)
    texts = []
    for ref_text, selected_text in pairs:
        texts.append(ref_text)
        texts.append(selected_text)

    concept_cache: dict[str, set[str]] = {}
    unique_texts = list(dict.fromkeys(texts))
    for doc in nlp.pipe(unique_texts, batch_size=batch_size):
        cuis = set()
        for ent in doc.ents:
            if ent._.kb_ents:
                cuis.add(ent._.kb_ents[0][0])
        concept_cache[doc.text] = cuis

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    jaccard_sum = 0.0
    empty_ref = 0
    for ref_text, selected_text in pairs:
        ref_cuis = concept_cache.get(ref_text, set())
        selected_cuis = concept_cache.get(selected_text, set())
        if not ref_cuis:
            empty_ref += 1
            precision = 1.0 if not selected_cuis else 0.0
            recall = 1.0
            jaccard = 1.0 if not selected_cuis else 0.0
        else:
            overlap = len(ref_cuis & selected_cuis)
            precision = overlap / len(selected_cuis) if selected_cuis else 0.0
            recall = overlap / len(ref_cuis)
            union = len(ref_cuis | selected_cuis)
            jaccard = overlap / union if union else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        jaccard_sum += jaccard

    count = len(pairs)
    return {
        "available": True,
        "model": model_name,
        "threshold": threshold,
        "count": count,
        "skipped": skipped,
        "empty_reference_concept_sets": empty_ref,
        "concept_precision": precision_sum / count,
        "concept_recall": recall_sum / count,
        "concept_f1": f1_sum / count,
        "concept_jaccard": jaccard_sum / count,
        "uco": jaccard_sum / count,
    }


def dataset_key(gt: dict[str, Any]) -> str:
    return str(gt.get("dataset_source") or gt.get("source") or "unknown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-length", action="store_true")
    parser.add_argument("--compute-uco", action="store_true")
    parser.add_argument("--uco-model", default="en_core_sci_sm")
    parser.add_argument("--uco-threshold", type=float, default=0.7)
    parser.add_argument("--uco-batch-size", type=int, default=64)
    parser.add_argument("--uco-text-mode", choices=["explanation", "selected_response"], default="explanation")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    ground_truth = normalize_gt(load_json(args.ground_truth))

    if args.strict_length and len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth)} labels")

    n = min(len(predictions), len(ground_truth))
    if args.max_samples is not None:
        n = min(n, args.max_samples)
    labels = []
    preds = []
    scores_b = []
    by_dataset = defaultdict(lambda: {"labels": [], "preds": [], "scores_b": []})
    invalid = 0

    for pred_row, gt_row in zip(predictions[:n], ground_truth[:n]):
        label = label_from_gt(gt_row)
        pred = prediction_label(pred_row)
        if label not in {"A", "B"}:
            continue
        if pred not in {"A", "B"}:
            invalid += 1
            pred = "B" if label == "A" else "A"
        score = score_for_b(pred_row, pred)
        labels.append(label)
        preds.append(pred)
        scores_b.append(score)

        bucket = by_dataset[dataset_key(gt_row)]
        bucket["labels"].append(label)
        bucket["preds"].append(pred)
        bucket["scores_b"].append(score)

    results = classification_metrics(labels, preds, scores_b)
    results["invalid_predictions"] = invalid
    results["coverage"] = {"predictions": len(predictions), "ground_truth": len(ground_truth), "evaluated": len(labels)}
    results["by_dataset"] = {
        key: classification_metrics(value["labels"], value["preds"], value["scores_b"])
        for key, value in sorted(by_dataset.items())
    }
    if args.compute_uco:
        results["uco"] = compute_uco(
            ground_truth[:n],
            predictions[:n],
            preds,
            model_name=args.uco_model,
            threshold=args.uco_threshold,
            batch_size=args.uco_batch_size,
            text_mode=args.uco_text_mode,
        )
    else:
        results["uco"] = {
            "available": False,
            "reason": "Pass --compute-uco to enable scispaCy/UMLS concept overlap.",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
