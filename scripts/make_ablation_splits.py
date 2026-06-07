#!/usr/bin/env python3
"""Create MedJudge leave-one-heuristic-out ablation splits.

The local data stores synthetic-pair heuristics in `candidate_type`. This script
creates LLaMA-Factory-ready train files with one candidate_type removed at a
time, while keeping the full test set for evaluation, matching the Table 3
ablation pattern.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_medjudge_llamafactory import convert_rm, convert_sft, load_json


DEFAULT_TYPES = ["random", "subtle_error", "kg_contradiction", "omission", "paraphrase_error"]


def dump_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def filter_without(raw_data: dict, candidate_type: str) -> dict:
    return {
        key: value
        for key, value in raw_data.items()
        if value.get("candidate_type") != candidate_type
    }


def dataset_info_entry(file_name: str, ranking: bool) -> dict:
    if ranking:
        return {
            "file_name": f"medjudge/{file_name}",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    return {
        "file_name": f"medjudge/{file_name}",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medthink-root", type=Path, default=Path("/path/to/Medthink"))
    parser.add_argument("--output-dir", type=Path, default=Path("/path/to/Medthink/LLaMA-Factory/data/medjudge"))
    parser.add_argument("--candidate-types", nargs="*", default=DEFAULT_TYPES)
    parser.add_argument("--keep-missing-images", action="store_true")
    args = parser.parse_args()

    final_merged = args.medthink_root / "Medthink_Dataset/Merged_Dataset/Pairwise_Dataset/Final_Merged"
    train_ab = load_json(final_merged / "train_all_pairwise_with_ab.json")
    train_expl = load_json(final_merged / "train_all_pairwise_with_explanation.json")
    test_ab = load_json(final_merged / "test_all_pairwise_with_ab.json")
    test_expl = load_json(final_merged / "test_all_pairwise_with_explanation.json")
    skip_missing = not args.keep_missing_images

    print("train candidate_type counts:", dict(Counter(v.get("candidate_type") for v in train_ab.values())))

    dataset_info = {}
    for candidate_type in args.candidate_types:
        safe = candidate_type.replace("-", "_")
        filtered_ab = filter_without(train_ab, candidate_type)
        filtered_expl = filter_without(train_expl, candidate_type)

        outputs = {
            f"medjudge_ablate_without_{safe}_sft_train.json": convert_sft(filtered_ab, False, skip_missing),
            f"medjudge_ablate_without_{safe}_sft_rea_train.json": convert_sft(filtered_expl, True, skip_missing),
            f"medjudge_ablate_without_{safe}_rm_train.json": convert_rm(filtered_ab, skip_missing),
            f"medjudge_ablate_without_{safe}_sft_test.json": convert_sft(test_ab, False, skip_missing),
            f"medjudge_ablate_without_{safe}_sft_rea_test.json": convert_sft(test_expl, True, skip_missing),
            f"medjudge_ablate_without_{safe}_rm_test.json": convert_rm(test_ab, skip_missing),
        }

        for file_name, data in outputs.items():
            dump_json(args.output_dir / file_name, data)
            key = file_name.removesuffix(".json")
            dataset_info[key] = dataset_info_entry(file_name, ranking="_rm_" in file_name)
            print(f"{file_name}: {len(data)}")

    info_path = args.output_dir / "ablation_dataset_info_snippet.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"dataset_info snippet: {info_path}")


if __name__ == "__main__":
    main()
