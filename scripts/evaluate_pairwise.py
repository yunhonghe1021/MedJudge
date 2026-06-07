#!/usr/bin/env python3
"""
Evaluation script for MedThink Pairwise predictions.
Computes:
1. Accuracy for A/B classification
2. BLEU score for prediction thinking vs ground truth solution
3. BERTScore for prediction thinking vs ground truth solution
"""

import json
import re
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def load_predictions(pred_path: str):
    """Load predictions from jsonl file."""
    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def load_ground_truth(gt_path: str):
    """Load ground truth from json file."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_answer(text: str) -> str:
    """Extract A or B from prediction text."""
    text = text.strip()

    # Check for final answer after </think>
    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        if after_think:
            text = after_think

    # Get last line or last character
    lines = text.strip().split("\n")
    last_line = lines[-1].strip() if lines else ""

    # Check last line first
    if last_line.upper() in ["A", "B"]:
        return last_line.upper()

    # Check if last line starts with A or B
    if last_line.upper().startswith("A"):
        return "A"
    if last_line.upper().startswith("B"):
        return "B"

    # Check entire text
    text_upper = text.upper()
    if "A" in text_upper and "B" not in text_upper:
        return "A"
    if "B" in text_upper and "A" not in text_upper:
        return "B"

    # Check last character
    if text and text[-1].upper() in ["A", "B"]:
        return text[-1].upper()

    return ""


def extract_thinking(text: str) -> str:
    """Extract thinking content from prediction."""
    # Try to extract content between <think> and </think>
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no tags, return everything before the final answer
    if "</think>" in text:
        return text.split("</think>")[0].replace("<think>", "").strip()

    # Return the whole text minus the last line (which is likely the answer)
    lines = text.strip().split("\n")
    if len(lines) > 1:
        return "\n".join(lines[:-1]).strip()

    return text.strip()


def compute_accuracy(predictions, ground_truth):
    """Compute accuracy for A/B classification."""
    correct = 0
    total = 0
    results_by_dataset = defaultdict(lambda: {"correct": 0, "total": 0})

    gt_list = list(ground_truth.values())

    for i, pred in enumerate(predictions):
        if i >= len(gt_list):
            break

        gt = gt_list[i]
        pred_answer = extract_answer(pred.get("predict", ""))
        gt_answer = gt.get("correct_answer", "").strip().upper()

        dataset_source = gt.get("dataset_source", "unknown")

        if pred_answer == gt_answer:
            correct += 1
            results_by_dataset[dataset_source]["correct"] += 1

        total += 1
        results_by_dataset[dataset_source]["total"] += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, results_by_dataset


def compute_bleu(predictions, ground_truth):
    """Compute BLEU score for thinking/explanation."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except ImportError:
        print("NLTK not available, skipping BLEU computation")
        return None

    smoothing = SmoothingFunction().method1
    gt_list = list(ground_truth.values())

    bleu_scores = []

    for i, pred in enumerate(tqdm(predictions, desc="Computing BLEU")):
        if i >= len(gt_list):
            break

        gt = gt_list[i]

        # Get prediction thinking
        pred_thinking = extract_thinking(pred.get("predict", ""))

        # Get ground truth solution
        gt_solution = gt.get("solution", "")

        if not pred_thinking or not gt_solution:
            continue

        # Tokenize
        try:
            pred_tokens = nltk.word_tokenize(pred_thinking.lower())
            gt_tokens = nltk.word_tokenize(gt_solution.lower())

            if len(pred_tokens) > 0 and len(gt_tokens) > 0:
                score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
        except Exception as e:
            continue

    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        return {
            "bleu_avg": avg_bleu,
            "bleu_count": len(bleu_scores),
        }
    return None


def compute_bertscore(predictions, ground_truth, batch_size=32):
    """Compute BERTScore for thinking/explanation."""
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("bert_score not available, skipping BERTScore computation")
        return None

    gt_list = list(ground_truth.values())

    pred_texts = []
    gt_texts = []

    for i, pred in enumerate(predictions):
        if i >= len(gt_list):
            break

        gt = gt_list[i]

        # Get prediction thinking
        pred_thinking = extract_thinking(pred.get("predict", ""))

        # Get ground truth solution
        gt_solution = gt.get("solution", "")

        if pred_thinking and gt_solution:
            pred_texts.append(pred_thinking)
            gt_texts.append(gt_solution)

    if not pred_texts:
        return None

    print(f"Computing BERTScore for {len(pred_texts)} samples...")

    # Compute BERTScore in batches
    P, R, F1 = bert_score(
        pred_texts,
        gt_texts,
        lang="en",
        verbose=True,
        batch_size=batch_size,
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
        "bertscore_count": len(pred_texts),
    }


def main():
    pred_path = "/path/to/Medthink/outputs/qwen3vl-4b-thinking/pairwise_sft/test_predictions.jsonl"
    gt_path = "/path/to/Medthink/Medthink_Dataset/Merged_Dataset/Pairwise_Dataset/Final_Merged/test_all_pairwise_with_ab.json"
    output_path = "/path/to/Medthink/outputs/qwen3vl-4b-thinking/pairwise_sft/evaluation_metrics.json"

    print("Loading predictions...")
    predictions = load_predictions(pred_path)
    print(f"Loaded {len(predictions)} predictions")

    print("Loading ground truth...")
    ground_truth = load_ground_truth(gt_path)
    print(f"Loaded {len(ground_truth)} ground truth samples")

    # Compute accuracy
    print("\n" + "=" * 70)
    print("Computing Accuracy...")
    print("=" * 70)
    accuracy, correct, total, results_by_dataset = compute_accuracy(predictions, ground_truth)
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("\nAccuracy by dataset:")
    for dataset, stats in sorted(results_by_dataset.items()):
        ds_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {dataset}: {ds_acc:.4f} ({stats['correct']}/{stats['total']})")

    # Compute BLEU
    print("\n" + "=" * 70)
    print("Computing BLEU Score...")
    print("=" * 70)
    bleu_results = compute_bleu(predictions, ground_truth)
    if bleu_results:
        print(f"BLEU Score: {bleu_results['bleu_avg']:.4f} (n={bleu_results['bleu_count']})")

    # Compute BERTScore
    print("\n" + "=" * 70)
    print("Computing BERTScore...")
    print("=" * 70)
    bertscore_results = compute_bertscore(predictions, ground_truth)
    if bertscore_results:
        print(f"BERTScore Precision: {bertscore_results['bertscore_precision']:.4f}")
        print(f"BERTScore Recall: {bertscore_results['bertscore_recall']:.4f}")
        print(f"BERTScore F1: {bertscore_results['bertscore_f1']:.4f}")

    # Save results
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_by_dataset": {
            k: {"accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0, **v}
            for k, v in results_by_dataset.items()
        },
    }

    if bleu_results:
        results.update(bleu_results)

    if bertscore_results:
        results.update(bertscore_results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
