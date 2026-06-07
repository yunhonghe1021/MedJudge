#!/usr/bin/env python3
"""Best-of-4 reranking for MedJudge.

Pipeline:
1. generate: run a VLM answer model four times per question to create candidates.
2. judge: compare all six candidate pairs with a MedJudge model and select the
   candidate with the most wins.

The judge stage matches the Table 2 setting conceptually: pairwise judge
decisions are used to rerank four generated answers into one final answer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PATH_PREFIXES = {
    "/path/to/Medthink/Medthink_Dataset/": "/path/to/Medthink/Medthink_Dataset/",
    "/path/to/Medthink/Medthink_Dataset/": "/path/to/Medthink/Medthink_Dataset/",
}


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        rows = []
        for idx, item in data.items():
            if isinstance(item, dict):
                item = {"id": str(idx), **item}
            rows.append(item)
        return rows
    if isinstance(data, list):
        return data
    raise TypeError(f"Unsupported input data type: {type(data)}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fix_image_path(path: str, dataset_root: Path) -> str:
    path = str(path)
    for old, new in PATH_PREFIXES.items():
        if path.startswith(old):
            path = new + path[len(old) :]
            break
    if os.path.exists(path):
        return path

    name = path
    source = ""
    if "dataset_source" in path:
        source = ""
    if not os.path.isabs(name):
        candidates = [
            dataset_root / "R-PathVQA" / "images" / "Test_images" / name,
            dataset_root / "R-PathVQA" / "images" / "Train_images" / name,
            dataset_root / "R-RAD" / "images" / name,
            dataset_root / "R-SLAKE" / "images" / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return path


def sample_image_path(item: dict[str, Any], dataset_root: Path) -> str:
    image = item.get("image") or item.get("img_name") or item.get("image_path")
    if not image and item.get("images"):
        image = item["images"][0]
    if not image:
        raise ValueError("Sample has no image field")
    return fix_image_path(str(image), dataset_root)


def answer_text(item: dict[str, Any]) -> str:
    answer = item.get("answer")
    choices = item.get("choices")
    if isinstance(answer, int) and isinstance(choices, list) and 0 <= answer < len(choices):
        return str(choices[answer])
    return str(answer)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"</?think>", " ", text)
    match = re.search(r"(?:the\s+answer\s+is|answer\s*:?)\s*([^.\n]+)", text)
    if match:
        text = match.group(1)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(gold)
    if not pred or not gold:
        return False
    return pred == gold or pred.startswith(gold) or gold in pred.split()


def extract_ab(text: str) -> str:
    text = str(text or "").strip()
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
        match = re.match(r"^[\(\[]?\s*([AB])\b", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def load_qwen3vl(model_name_or_path: str, adapter_path: str | None, flash_attention: bool):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.eval()
    return model, processor


def generate_one(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def make_answer_prompt(item: dict[str, Any]) -> str:
    question = item.get("question") or item.get("query") or ""
    choices = item.get("choices")
    if choices:
        choices_text = "\n".join(f"- {choice}" for choice in choices)
        return (
            f"Answer the medical visual question.\n\nQuestion: {question}\n\n"
            f"Choices:\n{choices_text}\n\nGive the final answer concisely."
        )
    return f"Answer the medical visual question.\n\nQuestion: {question}\n\nGive the final answer concisely."


def cmd_generate(args: argparse.Namespace) -> None:
    data = load_json_or_jsonl(args.input)
    if args.max_samples is not None:
        data = data[: args.max_samples]
    model, processor = load_qwen3vl(args.model, args.adapter, args.flash_attention)
    rows = []
    for index, item in enumerate(tqdm(data, desc="Generating 4 candidates")):
        try:
            image_path = sample_image_path(item, args.dataset_root)
            prompt = make_answer_prompt(item)
            candidates = []
            for k in range(args.num_candidates):
                candidates.append(
                    generate_one(
                        model,
                        processor,
                        image_path=image_path,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed + index * args.num_candidates + k if args.seed is not None else None,
                    )
                )
            rows.append(
                {
                    "id": str(item.get("id", index)),
                    "question": item.get("question"),
                    "image": image_path,
                    "answer": answer_text(item),
                    "dataset_source": item.get("dataset_source"),
                    "candidates": candidates,
                }
            )
        except Exception as exc:
            rows.append({"id": str(item.get("id", index)), "error": str(exc), "raw": item})
        if args.save_every and len(rows) % args.save_every == 0:
            write_jsonl(args.output, rows)
    write_jsonl(args.output, rows)


def make_judge_prompt(question: str, candidate_a: str, candidate_b: str) -> str:
    return (
        "You are a professional medical multimodal judge. Compare two candidate responses "
        "to the same medical visual question.\n\n"
        f"Question: {question}\n\n"
        f"Response A:\n{candidate_a}\n\n"
        f"Response B:\n{candidate_b}\n\n"
        "Which response is better? Answer with A or B."
    )


def cmd_judge(args: argparse.Namespace) -> None:
    data = load_json_or_jsonl(args.input)
    if args.max_samples is not None:
        data = data[: args.max_samples]
    model, processor = load_qwen3vl(args.judge_model, args.judge_adapter, args.flash_attention)
    rows = []
    correct = 0
    total = 0

    for item in tqdm(data, desc="Best-of-4 judge"):
        if item.get("error"):
            rows.append(item)
            continue
        candidates = item.get("candidates") or item.get("responses")
        if not isinstance(candidates, list) or len(candidates) < 4:
            rows.append({"id": item.get("id"), "error": "Expected at least 4 candidates", "raw": item})
            continue

        image_path = sample_image_path(item, args.dataset_root)
        wins = [0] * 4
        pairwise = []
        for i, j in itertools.combinations(range(4), 2):
            prompt = make_judge_prompt(str(item.get("question", "")), str(candidates[i]), str(candidates[j]))
            output = generate_one(
                model,
                processor,
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            decision = extract_ab(output)
            winner = None
            if decision == "A":
                winner = i
            elif decision == "B":
                winner = j
            if winner is not None:
                wins[winner] += 1
            pairwise.append({"i": i, "j": j, "decision": decision, "winner": winner, "output": output})

        best_index = max(range(4), key=lambda idx: (wins[idx], -idx))
        selected = candidates[best_index]
        gold = str(item.get("answer", ""))
        ok = is_correct(str(selected), gold)
        correct += int(ok)
        total += 1
        rows.append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "image": image_path,
                "answer": gold,
                "candidates": candidates[:4],
                "wins": wins,
                "selected_index": best_index,
                "selected": selected,
                "correct": ok,
                "pairwise": pairwise,
                "dataset_source": item.get("dataset_source"),
            }
        )
        if args.save_every and len(rows) % args.save_every == 0:
            write_jsonl(args.output, rows)

    write_jsonl(args.output, rows)
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "num_pairwise_judgments": total * 6,
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--input", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--dataset-root", type=Path, default=Path("/path/to/Medthink/Medthink_Dataset"))
    generate.add_argument("--model", required=True)
    generate.add_argument("--adapter")
    generate.add_argument("--num-candidates", type=int, default=4)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--max-samples", type=int)
    generate.add_argument("--save-every", type=int, default=20)
    generate.add_argument("--flash-attention", action="store_true")
    generate.set_defaults(func=cmd_generate)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--input", required=True, type=Path)
    judge.add_argument("--output", required=True, type=Path)
    judge.add_argument("--dataset-root", type=Path, default=Path("/path/to/Medthink/Medthink_Dataset"))
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--judge-adapter")
    judge.add_argument("--max-new-tokens", type=int, default=32)
    judge.add_argument("--temperature", type=float, default=0.0)
    judge.add_argument("--top-p", type=float, default=0.9)
    judge.add_argument("--max-samples", type=int)
    judge.add_argument("--save-every", type=int, default=20)
    judge.add_argument("--flash-attention", action="store_true")
    judge.set_defaults(func=cmd_judge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
