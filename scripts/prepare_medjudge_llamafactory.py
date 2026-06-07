#!/usr/bin/env python3
"""Prepare MedJudge datasets for the local January 2026 LLaMA-Factory checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM_PROMPT = """You are a professional medical multimodal judge. Compare two candidate responses to the same medical visual question.

Judge strictly by evidence consistency, localization accuracy, clinical reasonableness, semantic relevance, and appropriate granularity."""


PATH_PREFIXES = {
    "/path/to/Medthink/Medthink_Dataset/": "/path/to/Medthink/Medthink_Dataset/",
    "/path/to/Medthink/Medthink_Dataset/": "/path/to/Medthink/Medthink_Dataset/",
    "/path/to/Medthink/Medthink_Dataset/": "/path/to/Medthink/Medthink_Dataset/",
}


def fix_image_path(path: str) -> str:
    for old_prefix, new_prefix in PATH_PREFIXES.items():
        if path.startswith(old_prefix):
            return path.replace(old_prefix, new_prefix, 1)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_pairwise_prompt(item: dict, response_a: str | None = None, response_b: str | None = None) -> str:
    response_a = item.get("A", item.get("candidate", "")) if response_a is None else response_a
    response_b = item.get("B", item.get("reference", item.get("output", ""))) if response_b is None else response_b
    return f"""<image>Question: {item.get("question", "")}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Answer with A or B."""


def make_single_response_prompt(item: dict) -> str:
    return f"""<image>Question: {item.get("question", "")}

Evaluate the candidate response for this medical image question."""


def make_reasoning_answer(item: dict, answer: str) -> str:
    explanation = item.get("explanation", "").strip()
    if explanation:
        return f"{explanation}\n\nFinal preference: {answer}"
    return f"Final preference: {answer}"


def get_ab(item: dict) -> tuple[str, str, str]:
    answer = item.get("correct_answer", "").strip().upper()
    response_a = item.get("A", "")
    response_b = item.get("B", "")
    if answer == "A":
        return response_a, response_b, answer
    if answer == "B":
        return response_b, response_a, answer
    raise ValueError(f"Invalid correct_answer: {answer!r}")


def place_reference_candidate(key: str, item: dict) -> tuple[str, str, str]:
    """Return response A, response B, and the correct label for explanation data.

    The source explanation data stores `reference` as preferred and `candidate`
    as rejected. We deterministically flip half of the samples by id to avoid
    teaching a positional shortcut.
    """
    reference = item.get("reference", item.get("output", ""))
    candidate = item.get("candidate", "")
    digest = hashlib.md5(str(key).encode("utf-8")).digest()
    reference_is_a = digest[0] % 2 == 0
    if reference_is_a:
        return reference, candidate, "A"
    return candidate, reference, "B"


def convert_sft(raw_data: dict, with_reasoning: bool, skip_missing_images: bool) -> list[dict]:
    converted = []
    for key, item in raw_data.items():
        image = fix_image_path(item.get("image", ""))
        if skip_missing_images and not Path(image).is_file():
            continue
        if with_reasoning:
            response_a, response_b, answer = place_reference_candidate(str(key), item)
        else:
            response_a, response_b = item.get("A", ""), item.get("B", "")
            answer = item.get("correct_answer", "").strip().upper()
            if answer not in {"A", "B"}:
                continue

        converted.append(
            {
                "id": str(key),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_pairwise_prompt(item, response_a, response_b)},
                    {"role": "assistant", "content": make_reasoning_answer(item, answer) if with_reasoning else answer},
                ],
                "images": [image],
            }
        )
    return converted


def convert_rm(raw_data: dict, skip_missing_images: bool) -> list[dict]:
    converted = []
    for key, item in raw_data.items():
        image = fix_image_path(item.get("image", ""))
        if skip_missing_images and not Path(image).is_file():
            continue
        try:
            chosen, rejected, _answer = get_ab(item)
        except ValueError:
            continue

        converted.append(
            {
                "id": str(key),
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_single_response_prompt(item)},
                ],
                "chosen": {"role": "assistant", "content": chosen},
                "rejected": {"role": "assistant", "content": rejected},
                "images": [image],
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medthink-root", type=Path, default=Path("/path/to/Medthink"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/Medthink/LLaMA-Factory/data/medjudge"),
    )
    parser.add_argument("--keep-missing-images", action="store_true")
    args = parser.parse_args()

    final_merged = (
        args.medthink_root
        / "Medthink_Dataset/Merged_Dataset/Pairwise_Dataset/Final_Merged"
    )
    train_ab = load_json(final_merged / "train_all_pairwise_with_ab.json")
    test_ab = load_json(final_merged / "test_all_pairwise_with_ab.json")
    train_expl = load_json(final_merged / "train_all_pairwise_with_explanation.json")
    test_expl = load_json(final_merged / "test_all_pairwise_with_explanation.json")

    skip_missing = not args.keep_missing_images
    outputs = {
        "medjudge_sft_train.json": convert_sft(train_ab, with_reasoning=False, skip_missing_images=skip_missing),
        "medjudge_sft_test.json": convert_sft(test_ab, with_reasoning=False, skip_missing_images=skip_missing),
        "medjudge_sft_rea_train.json": convert_sft(
            train_expl, with_reasoning=True, skip_missing_images=skip_missing
        ),
        "medjudge_sft_rea_test.json": convert_sft(test_expl, with_reasoning=True, skip_missing_images=skip_missing),
        "medjudge_rm_train.json": convert_rm(train_ab, skip_missing_images=skip_missing),
        "medjudge_rm_test.json": convert_rm(test_ab, skip_missing_images=skip_missing),
    }

    for name, data in outputs.items():
        dump_json(args.output_dir / name, data)
        print(f"{name}: {len(data)}")


if __name__ == "__main__":
    main()
