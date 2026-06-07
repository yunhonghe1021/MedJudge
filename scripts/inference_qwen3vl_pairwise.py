#!/usr/bin/env python3
"""
Inference script for Qwen3-VL-4B-Thinking on MedThink Pairwise Test Dataset
Using HuggingFace Transformers with batch inference (no vLLM)
"""

import json
import os
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image


def load_model_and_processor(
    model_name_or_path: str,
    adapter_path: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load model and processor with optional LoRA adapter."""
    print(f"Loading model from {model_name_or_path}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    return model, processor


def load_test_dataset(dataset_path: str):
    """Load test dataset in sharegpt format."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


def process_batch(
    model,
    processor,
    batch_samples,
    max_new_tokens: int = 16,
    temperature: float = 0.1,
    top_p: float = 0.9,
):
    """Process a batch of samples."""
    batch_messages = []
    batch_labels = []
    batch_prompts = []
    valid_indices = []

    for idx, sample in enumerate(batch_samples):
        try:
            user_content = sample["messages"][0]["content"]
            label = sample["messages"][1]["content"]
            image_path = sample["images"][0]

            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_content.replace("<image>", "").strip()},
                    ],
                }
            ]

            batch_messages.append(messages)
            batch_labels.append(label)
            batch_prompts.append(user_content)
            valid_indices.append(idx)

        except Exception as e:
            print(f"Error loading sample: {e}")
            continue

    if not batch_messages:
        return []

    # Process batch
    try:
        # For Qwen3-VL, we need to process each sample individually due to variable image sizes
        # But we can still batch the generation
        batch_inputs = []
        for messages in batch_messages:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs.append(inputs)

        # Pad to same length
        max_len = max(inp["input_ids"].shape[1] for inp in batch_inputs)

        padded_input_ids = []
        padded_attention_mask = []
        pixel_values_list = []
        image_grid_thw_list = []

        for inp in batch_inputs:
            seq_len = inp["input_ids"].shape[1]
            pad_len = max_len - seq_len

            # Pad input_ids (left padding for generation)
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), processor.tokenizer.pad_token_id, dtype=inp["input_ids"].dtype)
                padded_input_ids.append(torch.cat([pad_ids, inp["input_ids"]], dim=1))
                pad_mask = torch.zeros((1, pad_len), dtype=inp["attention_mask"].dtype)
                padded_attention_mask.append(torch.cat([pad_mask, inp["attention_mask"]], dim=1))
            else:
                padded_input_ids.append(inp["input_ids"])
                padded_attention_mask.append(inp["attention_mask"])

            if "pixel_values" in inp:
                pixel_values_list.append(inp["pixel_values"])
            if "image_grid_thw" in inp:
                image_grid_thw_list.append(inp["image_grid_thw"])

        # Stack tensors
        batched_inputs = {
            "input_ids": torch.cat(padded_input_ids, dim=0).to(model.device),
            "attention_mask": torch.cat(padded_attention_mask, dim=0).to(model.device),
        }

        if pixel_values_list:
            batched_inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0).to(model.device)
        if image_grid_thw_list:
            batched_inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **batched_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode outputs
        results = []
        for i, (gen_ids, prompt, label) in enumerate(zip(generated_ids, batch_prompts, batch_labels)):
            # Find where the generated tokens start
            input_len = batched_inputs["input_ids"].shape[1]
            generated_ids_trimmed = gen_ids[input_len:]

            output_text = processor.tokenizer.decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            results.append({
                "prompt": prompt,
                "predict": output_text.strip(),
                "label": label,
            })

        return results

    except Exception as e:
        print(f"Batch processing error: {e}")
        # Fallback to single sample processing
        results = []
        for messages, prompt, label in zip(batch_messages, batch_prompts, batch_labels):
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )

                generated_ids_trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
                output_text = processor.tokenizer.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                results.append({
                    "prompt": prompt,
                    "predict": output_text.strip(),
                    "label": label,
                })
            except Exception as e2:
                results.append({
                    "prompt": prompt,
                    "predict": "",
                    "label": label,
                    "error": str(e2),
                })

        return results


def run_inference(
    model,
    processor,
    dataset,
    output_path: str,
    max_new_tokens: int = 16,
    temperature: float = 0.1,
    top_p: float = 0.9,
    batch_size: int = 32,
):
    """Run batched inference on the dataset."""
    results = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_samples = dataset[start_idx:end_idx]

        batch_results = process_batch(
            model,
            processor,
            batch_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        results.extend(batch_results)

        # Save intermediate results every 10 batches
        if (batch_idx + 1) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\nSaved {len(results)} results to {output_path}")

    # Save final results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {output_path}")
    return results


def compute_accuracy(results):
    """Compute accuracy for pairwise classification."""
    correct = 0
    total = 0

    for r in results:
        if "error" in r:
            continue

        pred = r["predict"].strip().upper()
        label = r["label"].strip().upper()

        # Extract just A or B from prediction
        if "A" in pred and "B" not in pred:
            pred = "A"
        elif "B" in pred and "A" not in pred:
            pred = "B"
        elif pred.startswith("A"):
            pred = "A"
        elif pred.startswith("B"):
            pred = "B"

        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def main():
    # Configuration
    model_name_or_path = "Qwen/Qwen3-VL-4B-Thinking"
    adapter_path = "/path/to/Medthink/outputs/qwen3vl-4b-thinking/pairwise_sft/checkpoint-1069"
    dataset_path = "/path/to/Medthink/Medthink_Dataset/Merged_Dataset/Pairwise_Dataset/Final_Merged/test_pairwise_sharegpt.json"
    output_path = "/path/to/Medthink/outputs/qwen3vl-4b-thinking/pairwise_sft/test_predictions.jsonl"

    max_new_tokens = 1024  # Increased for thinking output
    temperature = 0  # Higher temperature for thinking
    top_p = 0.9
    batch_size = 32

    # Load model
    model, processor = load_model_and_processor(
        model_name_or_path,
        adapter_path=adapter_path,
    )

    # Load dataset
    dataset = load_test_dataset(dataset_path)

    # Run inference
    results = run_inference(
        model,
        processor,
        dataset,
        output_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )

    # Compute accuracy
    accuracy, correct, total = compute_accuracy(results)
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("=" * 70)

    # Save metrics
    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
