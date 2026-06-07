#!/usr/bin/env python3
"""
Convert test pairwise dataset to LLaMA-Factory sharegpt format for Qwen3-VL inference.
"""

import json

# Path mapping: old path prefix -> new path prefix
PATH_MAPPING = {
    "/path/to/Medthink/Medthink_Dataset/": "/path/to/Medthink/Medthink_Dataset/",
}


def fix_image_path(image_path: str) -> str:
    """Replace old path prefixes with new ones."""
    for old_prefix, new_prefix in PATH_MAPPING.items():
        if image_path.startswith(old_prefix):
            return image_path.replace(old_prefix, new_prefix, 1)
    return image_path


def convert_pairwise_to_sharegpt(input_path: str, output_path: str):
    """Convert pairwise dataset to sharegpt format."""

    print(f"Loading dataset from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_data = []

    print(f"Converting {len(data)} entries...")
    for key, item in data.items():
        question = item.get("question", "")
        image_path = item.get("image", "")
        response_a = item.get("A", "")
        response_b = item.get("B", "")
        correct_answer = item.get("correct_answer", "")

        # Skip if missing essential fields
        if not question or not image_path or not response_a or not response_b or not correct_answer:
            continue

        # Fix image path
        image_path = fix_image_path(image_path)

        # Build the user prompt for pairwise comparison
        user_content = f"""<image>Given the medical image and the following question:

Question: {question}

Here are two candidate responses:

**Response A:**
{response_a}

**Response B:**
{response_b}

Based on the image and question, which response is more accurate and appropriate? Please answer with just 'A' or 'B'."""

        # The assistant response is just the correct answer letter
        assistant_content = correct_answer

        converted_entry = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "images": [image_path]
        }

        converted_data.append(converted_entry)

    print(f"Converted {len(converted_data)} entries")

    # Save to output file
    print(f"Saving to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print("Done!")
    return len(converted_data)


if __name__ == "__main__":
    input_file = "/path/to/Medthink/Medthink_Dataset/Merged_Dataset/Pairwise_Dataset/Final_Merged/test_all_pairwise_with_ab.json"
    output_file = "/path/to/Medthink/Medthink_Dataset/Merged_Dataset/Pairwise_Dataset/Final_Merged/test_pairwise_sharegpt.json"

    convert_pairwise_to_sharegpt(input_file, output_file)
